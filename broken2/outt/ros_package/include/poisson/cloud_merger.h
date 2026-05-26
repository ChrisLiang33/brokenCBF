//good

// cloud_merger.h
// LiDAR point cloud to occupancy grid converter (camera-free version).
// Stripped out:
//   - RealSense/ZED camera point cloud subscription and callback
//   - All camera-related includes and processing
// Kept:
//   - Livox Mid360 LiDAR subscription (/livox/lidar)
//   - Go2 front UTLidar subscription (/utlidar/cloud)
//   - TF2-based point cloud transforms
//   - Occupancy grid generation with Gaussian confidence filtering
//   - Robot body masking (hyper-ellipse filter)
//
// Publishes:
//   - "occupancy_grid" (nav_msgs/OccupancyGrid) — confidence-based occupancy
//   - "poisson_cloud" (sensor_msgs/PointCloud2) — filtered point cloud
#pragma once

#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>
#include <Eigen/Dense>
#include "utils.h"
#include "grid_params.h"

bool initialized = false;

const float minX = 0.40f;  // Robot body mask X (must be >= 0.370)
const float maxX = (float)(JMAX / 2) * DS;
const float minY = 0.20f;  // Robot body mask Y (must be >= 0.185)
const float maxY = (float)(IMAX / 2) * DS;
const float minZ_default = 0.05f;
const float maxZ_default = 0.80f;

class CloudMergerNode : public rclcpp::Node {
public:
    CloudMergerNode(float min_z_override = -1.0f, float max_z_override = -1.0f)
        : Node("cloud_merger") {
        // Use overrides if provided, otherwise use ROS parameter or default
        this->declare_parameter("min_z", minZ_default);
        this->declare_parameter("max_z", maxZ_default);
        if (min_z_override >= 0.0f)
            minZ_ = min_z_override;
        else
            minZ_ = this->get_parameter("min_z").as_double();
        if (max_z_override > 0.0f)
            maxZ_ = max_z_override;
        else
            maxZ_ = this->get_parameter("max_z").as_double();
        RCLCPP_INFO(this->get_logger(), "CloudMerger min_z=%.2f, max_z=%.2f", minZ_, maxZ_);

        // Initialize Cloud Message
        cloud_msg.header.stamp = this->now();
        cloud_msg.header.frame_id = "body_link";

        // Initialize Map Message
        map_msg.data.resize(IMAX * JMAX);
        map_msg.header.stamp = this->now();
        map_msg.header.frame_id = "body_link";
        map_msg.info.width = IMAX;
        map_msg.info.height = JMAX;
        map_msg.info.resolution = DS;
        map_msg.info.origin.position.x = -maxX;
        map_msg.info.origin.position.y = -maxY;
        map_msg.info.origin.position.z = 0.0f;
        map_msg.info.origin.orientation.w = 1.0;
        map_msg.info.origin.orientation.x = 0.0f;
        map_msg.info.origin.orientation.y = 0.0f;
        map_msg.info.origin.orientation.z = 0.0f;

        // Construct Initial Grids
        for (int i = 0; i < IMAX; i++) {
            for (int j = 0; j < JMAX; j++) {
                const float x = (float)(j - JMAX / 2) * DS;
                const float y = (float)(i - IMAX / 2) * DS;
                polar_coordinates_r2[i * JMAX + j] = x * x + y * y;
                polar_coordinates_th[i * JMAX + j] = std::atan2(y, x);
                old_conf[i * JMAX + j] = 0;
            }
        }

        // Start Time
        t = std::chrono::steady_clock::now();

        // Create Subscribers & Publishers
        livox_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar", 1,
            std::bind(&CloudMergerNode::lidar_callback, this, std::placeholders::_1));
        utlidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/utlidar/cloud", 1,
            std::bind(&CloudMergerNode::combined_callback, this, std::placeholders::_1));
        cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("poisson_cloud", 1);
        map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1);

        // TF2 setup for transforms
        target_frame_ = "body_link";
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        combined_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    }

private:
    void combined_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        Timer map_timer(true);
        map_timer.start();

        dt = std::chrono::duration<float>(std::chrono::steady_clock::now() - t).count();
        t = std::chrono::steady_clock::now();

        pcl::PointCloud<pcl::PointXYZI>::Ptr odom_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *odom_cloud);

        if (!transform_pointcloud(odom_cloud, msg->header.frame_id)) return;

        *odom_cloud += *combined_cloud_;
        combined_cloud_->clear();

        // Create Occupancy Grid
        cv::Mat raw_map = cv::Mat::zeros(IMAX, JMAX, CV_32F);
        for (const auto& pt : odom_cloud->points) {
            const bool in_plane = (pt.z > minZ_) && (pt.z < maxZ_);
            if (!in_plane) continue;
            const float ic = pt.y / DS + (float)(IMAX / 2);
            const float jc = pt.x / DS + (float)(JMAX / 2);
            const bool in_grid = (ic > 0.0f) && (ic < (float)(IMAX - 1)) &&
                                 (jc > 0.0f) && (jc < (float)(JMAX - 1));
            if (!in_grid) continue;
            raw_map.at<float>((int)std::round(ic), (int)std::round(jc)) = 1.0f;
        }

        // Build confidence map
        for (int n = 0; n < IMAX * JMAX; n++) confidence_values[n] = 0;
        Filtered_Occupancy_Convolution(confidence_values, raw_map, old_conf);
        memcpy(old_conf, confidence_values, IMAX * JMAX * sizeof(int8_t));

        // Publish filtered cloud
        pcl::toROSMsg(*odom_cloud, cloud_msg);
        cloud_msg.header.stamp = this->now();
        cloud_msg.header.frame_id = "body_link";
        cloud_pub_->publish(cloud_msg);

        // Publish occupancy grid
        for (int n = 0; n < IMAX * JMAX; n++) map_msg.data[n] = confidence_values[n];
        map_msg.header.stamp = this->now();
        map_msg.info.origin.position.x = -maxX;
        map_msg.info.origin.position.y = -maxY;
        map_pub_->publish(map_msg);

        // Throttle timing prints to ~1Hz
        static int timing_print_counter = 0;
        if (++timing_print_counter >= 15) {
            timing_print_counter = 0;
            map_timer.time("Occ Map Solve Time: ");
        }
    }

    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!update_pose_from_tf()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for TF odom -> body_link, skipping lidar frame");
            return;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);

        if (!transform_pointcloud(cloud, msg->header.frame_id)) return;

        // Mask robot body with hyper-ellipse
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto& pt : cloud->points) {
            float ellipse_norm = std::pow(pt.x / minX, 8.0f) + std::pow(pt.y / minY, 8.0f);
            if (ellipse_norm > 1.0f) filtered->points.push_back(pt);
        }
        filtered->width = filtered->points.size();
        filtered->height = 1;

        *combined_cloud_ += *filtered;
    }

    bool update_pose_from_tf() {
        try {
            if (!tf_buffer_->canTransform("odom", "body_link", tf2::TimePointZero))
                return false;
            auto transform = tf_buffer_->lookupTransform(
                "odom", "body_link", tf2::TimePointZero, tf2::durationFromSec(0.05));
            r[0] = transform.transform.translation.x;
            r[1] = transform.transform.translation.y;
            r[2] = transform.transform.translation.z;
            return true;
        } catch (tf2::TransformException&) {
            return false;
        }
    }

    bool transform_pointcloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                              const std::string& source_frame) {
        if (source_frame == target_frame_) return true;
        if (source_frame.empty()) return false;
        if (!tf_buffer_) return false;
        try {
            if (!tf_buffer_->canTransform(target_frame_, source_frame, tf2::TimePointZero)) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Waiting for TF from %s to %s...", source_frame.c_str(), target_frame_.c_str());
                return false;
            }
            geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
                target_frame_, source_frame, tf2::TimePointZero, tf2::durationFromSec(0.1));
            Eigen::Affine3d eigen_transform = tf2::transformToEigen(transform.transform);
            pcl::transformPointCloud(*cloud, *cloud, eigen_transform.cast<float>());
            return true;
        } catch (std::exception& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "TF transform failed: %s", ex.what());
            return false;
        }
    }

    cv::Mat gaussian_kernel(int kernel_size, float sigma) {
        cv::Mat kernel(kernel_size, kernel_size, CV_32F);
        int half = kernel_size / 2;
        for (int i = -half; i <= half; i++) {
            for (int j = -half; j <= half; j++) {
                float val = std::exp(-(i * i + j * j) / (2.0 * sigma * sigma));
                kernel.at<float>(i + half, j + half) = val;
            }
        }
        return kernel;
    }

    void Filtered_Occupancy_Convolution(int8_t* confidence_values,
                                        const cv::Mat& occupancy_data,
                                        const int8_t* old_conf_map) {
        for (int i = 0; i < IMAX; i++) {
            for (int j = 0; j < JMAX; j++) {
                confidence_values[i * JMAX + j] = old_conf_map[i * JMAX + j];
            }
        }

        cv::filter2D(occupancy_data, buffered_binary, -1, gauss_kernel,
                      cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

        float sig, C, beta_up, beta_dn;
        const float thresh_front = 2.0f;
        const float thresh_mid360 = 4.0f;
        bool front_flag = true;

        for (int i = 0; i < IMAX; i++) {
            for (int j = 0; j < JMAX; j++) {
                const float r2 = polar_coordinates_r2[i * JMAX + j];
                const float th = polar_coordinates_th[i * JMAX + j];
                const bool range_flag = r2 > 1.44f;
                const bool angle_flag = std::abs(ang_diff(0.0f, th)) > 0.6f;
                if (range_flag || angle_flag) front_flag = false;
                else front_flag = true;

                const float thresh = front_flag ? thresh_front : thresh_mid360;

                float val_binary = buffered_binary.at<float>(i, j);
                float conf = (float)confidence_values[i * JMAX + j] / 127.0f;
                if (val_binary > thresh) {
                    if (front_flag) beta_up = 4.0f;
                    else beta_up = 1.0f;
                    sig = 1.0f - std::exp(-beta_up * val_binary * dt);
                    C = 1.0f;
                } else {
                    if (front_flag) beta_dn = 4.0f;
                    else beta_dn = 4.0f;
                    sig = 1.0f - std::exp(-beta_dn * dt);
                    C = 0.0f;
                }
                conf *= 1.0f - sig;
                conf += sig * C;
                confidence_values[i * JMAX + j] = (int8_t)std::round(127.0f * conf);
            }
        }
    }

    sensor_msgs::msg::PointCloud2 cloud_msg;
    nav_msgs::msg::OccupancyGrid map_msg;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr livox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr utlidar_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::string target_frame_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr combined_cloud_;

    float minZ_ = minZ_default;
    float maxZ_ = maxZ_default;

    std::vector<float> r = {0.0f, 0.0f, 0.0f};
    std::chrono::steady_clock::time_point t;
    float dt = 1.0e10f;

    const cv::Mat gauss_kernel = gaussian_kernel(9, 2.0);

    int8_t confidence_values[IMAX * JMAX];
    int8_t old_conf[IMAX * JMAX];
    float polar_coordinates_r2[IMAX * JMAX];
    float polar_coordinates_th[IMAX * JMAX];
    cv::Mat buffered_binary = cv::Mat::zeros(IMAX, JMAX, CV_32F);
};
