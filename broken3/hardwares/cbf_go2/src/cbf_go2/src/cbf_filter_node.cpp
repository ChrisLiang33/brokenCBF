// cbf_filter_node.cpp
//
// Closed-form 2D CBF QP filter. Sits between teleop and walking_bridge:
//   /u_teleop   (raw user cmd, geometry_msgs/Twist)
//   /cbf/params (alpha, phi from cbf_inference_node, std_msgs/Float32MultiArray)
//   /poisson_cloud (filtered LiDAR points from cloud_merger, sensor_msgs/PointCloud2)
//   ->
//   /u_des      (filtered safe command, geometry_msgs/Twist) -> walking_bridge
//
// CBF constraint (matches cbf_qp.py / cbf_go2_env._cbf_filter for the MVP
// case where a, b, c are pinned to 0):
//
//   L_g h . u >= -alpha * h  +  phi * ||L_g h||^2
//
//   h(x)   = ||robot - nearest_obs|| - r_safety
//   L_g h  = unit vector from nearest obstacle to robot (gradient of h
//            w.r.t. robot xy; in body frame, robot is at origin and the
//            obstacle is at p_nearest, so L_g h = -p_nearest_hat).
//
// Single linear inequality in 2D -> closed-form half-space projection
// (no QP solver needed). This is the same projection the sim uses.
//
// Ported from hardwares/prototype/deploy/cbf_filter_node.py (V14). The
// Python version supported (alpha, phi, a, b, c); we drop a/b/c to match
// the MVP's (phi, alpha) action space and re-introduce them only if the
// trained policy starts emitting them.
//
// Fail-safes:
//   - Stale params (>200ms): pass u_teleop through unchanged.
//   - Stale LiDAR (>300ms): pass u_teleop through unchanged.
//   - NaN/Inf on inputs: zero the output (walking_bridge's heartbeat
//     watchdog will dampen if this persists).
//   - L_g h ~ 0 but constraint binding: emergency stop (zero output).
//
// Build target: registered in CMakeLists.txt as `cbf_filter_node`.

#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>


// Safety geometry — must match the sim's _cbf_filter constants.
constexpr float R_ROBOT          = 0.40f;
constexpr float R_OBS_NOMINAL    = 0.30f;
constexpr float SAFETY_MARGIN    = R_ROBOT + R_OBS_NOMINAL;  // 0.70 m

// LiDAR -> body-frame obstacle filter.
constexpr float Z_MIN_OBSTACLE   = 0.05f;
constexpr float Z_MAX_OBSTACLE   = 0.80f;
constexpr float BODY_MASK_X      = 0.40f;
constexpr float BODY_MASK_Y      = 0.20f;

// Default tick rate. Matches sim physics step (50 Hz).
constexpr double DEFAULT_RATE_HZ = 50.0;

// Staleness windows.
constexpr double PARAMS_STALE_S  = 0.2;
constexpr double LIDAR_STALE_S   = 0.3;
constexpr double TELEOP_STALE_S  = 0.5;


class CbfFilterNode : public rclcpp::Node {
public:
    CbfFilterNode() : Node("cbf_filter_node"),
                      tf_buffer_(this->get_clock()),
                      tf_listener_(tf_buffer_) {
        // Parameters.
        this->declare_parameter("cloud_topic", "/poisson_cloud");
        this->declare_parameter("body_frame", "body_link");
        this->declare_parameter("safety_margin", static_cast<double>(SAFETY_MARGIN));
        this->declare_parameter("publish_rate_hz", DEFAULT_RATE_HZ);

        cloud_topic_   = this->get_parameter("cloud_topic").as_string();
        body_frame_    = this->get_parameter("body_frame").as_string();
        safety_margin_ = static_cast<float>(this->get_parameter("safety_margin").as_double());
        double rate_hz = this->get_parameter("publish_rate_hz").as_double();

        // QoS — sensor data best-effort, commands reliable.
        auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
        auto cmd_qos    = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();

        teleop_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/u_teleop", cmd_qos,
            std::bind(&CbfFilterNode::teleop_cb, this, std::placeholders::_1));
        params_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/cbf/params", cmd_qos,
            std::bind(&CbfFilterNode::params_cb, this, std::placeholders::_1));
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_, sensor_qos,
            std::bind(&CbfFilterNode::cloud_cb, this, std::placeholders::_1));

        u_des_pub_  = this->create_publisher<geometry_msgs::msg::Twist>("/u_des", cmd_qos);
        status_pub_ = this->create_publisher<std_msgs::msg::String>("/cbf/filter_status", 1);
        h_pub_      = this->create_publisher<std_msgs::msg::Float32MultiArray>("/cbf/filter_h", sensor_qos);

        const auto tick_period = std::chrono::duration<double>(1.0 / rate_hz);
        tick_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tick_period),
            std::bind(&CbfFilterNode::tick, this));
        heartbeat_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&CbfFilterNode::heartbeat, this));

        RCLCPP_INFO(this->get_logger(),
                    "CbfFilterNode: cloud_topic=%s, body_frame=%s, safety_margin=%.2fm, rate=%.1fHz",
                    cloud_topic_.c_str(), body_frame_.c_str(), safety_margin_, rate_hz);
    }

private:
    // ---- Callbacks ----

    void teleop_cb(const geometry_msgs::msg::Twist::SharedPtr msg) {
        last_teleop_time_ = this->now();
        teleop_vx_  = static_cast<float>(msg->linear.x);
        teleop_vy_  = static_cast<float>(msg->linear.y);
        teleop_wyaw_ = static_cast<float>(msg->angular.z);
    }

    void params_cb(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        // MVP layout: [alpha, phi]. Accept >= 2 entries; ignore tail
        // (forward-compat with the prototype's 5-dim (alpha, phi, a, b, c)).
        if (msg->data.size() < 2) {
            RCLCPP_WARN(this->get_logger(),
                        "/cbf/params has %zu entries, expected >= 2 [alpha, phi]",
                        msg->data.size());
            return;
        }
        last_params_time_ = this->now();
        alpha_ = msg->data[0];
        phi_   = msg->data[1];
    }

    void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        last_cloud_time_ = this->now();
        latest_cloud_ = msg;
    }

    // ---- Geometry ----

    // Find the nearest in-plane obstacle point in body frame; populate
    // h and L_g h. Returns false if no usable points (caller passes
    // through). Sets h = +inf if all points are out of range.
    bool compute_h_and_grad(float& h, float& Lgh_x, float& Lgh_y) {
        if (!latest_cloud_) return false;
        const auto& cloud = *latest_cloud_;

        // Resolve transform cloud_frame -> body_frame.
        float R[9] = {1,0,0, 0,1,0, 0,0,1};
        float tx = 0.f, ty = 0.f, tz = 0.f;
        const bool needs_tf = (cloud.header.frame_id != body_frame_);
        if (needs_tf) {
            geometry_msgs::msg::TransformStamped tf;
            try {
                tf = tf_buffer_.lookupTransform(
                    body_frame_, cloud.header.frame_id, cloud.header.stamp,
                    rclcpp::Duration::from_seconds(0.05));
            } catch (const tf2::TransformException& exc) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "TF %s -> %s failed: %s",
                                     cloud.header.frame_id.c_str(),
                                     body_frame_.c_str(), exc.what());
                return false;
            }
            const float qw = tf.transform.rotation.w;
            const float qx = tf.transform.rotation.x;
            const float qy = tf.transform.rotation.y;
            const float qz = tf.transform.rotation.z;
            const float xx = qx*qx, yy = qy*qy, zz = qz*qz;
            const float xy = qx*qy, xz = qx*qz, yz = qy*qz;
            const float wx = qw*qx, wy = qw*qy, wz_ = qw*qz;
            R[0] = 1.f - 2.f*(yy + zz);  R[1] = 2.f*(xy - wz_);    R[2] = 2.f*(xz + wy);
            R[3] = 2.f*(xy + wz_);       R[4] = 1.f - 2.f*(xx+zz); R[5] = 2.f*(yz - wx);
            R[6] = 2.f*(xz - wy);        R[7] = 2.f*(yz + wx);     R[8] = 1.f - 2.f*(xx+yy);
            tx = static_cast<float>(tf.transform.translation.x);
            ty = static_cast<float>(tf.transform.translation.y);
            tz = static_cast<float>(tf.transform.translation.z);
        }

        // Iterate and track minimum |xy|^2 in body frame.
        sensor_msgs::PointCloud2ConstIterator<float> ix(cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iy(cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iz(cloud, "z");

        float best_d2 = std::numeric_limits<float>::infinity();
        float best_bx = 0.f, best_by = 0.f;
        bool any_valid = false;
        bool any_in_range = false;

        for (; ix != ix.end(); ++ix, ++iy, ++iz) {
            const float px = *ix, py = *iy, pz = *iz;
            if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) continue;

            // To body frame.
            float bx, by, bz;
            if (needs_tf) {
                bx = R[0]*px + R[1]*py + R[2]*pz + tx;
                by = R[3]*px + R[4]*py + R[5]*pz + ty;
                bz = R[6]*px + R[7]*py + R[8]*pz + tz;
            } else {
                bx = px; by = py; bz = pz;
            }
            any_valid = true;

            // Z filter + body footprint mask.
            if (bz < Z_MIN_OBSTACLE || bz > Z_MAX_OBSTACLE) continue;
            if (std::fabs(bx) < BODY_MASK_X && std::fabs(by) < BODY_MASK_Y) continue;

            const float d2 = bx*bx + by*by;
            any_in_range = true;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_bx = bx;
                best_by = by;
            }
        }

        if (!any_valid) return false;
        if (!any_in_range) {
            // No obstacles to filter against. Mark h as +inf so the
            // caller short-circuits to passthrough.
            h = std::numeric_limits<float>::infinity();
            Lgh_x = 0.f; Lgh_y = 0.f;
            return true;
        }

        const float d = std::sqrt(best_d2);
        h = d - safety_margin_;
        // In body frame, robot at origin -> grad of ||robot - obs|| w.r.t.
        // robot xy is (robot - obs) / ||...|| = -p_nearest_hat.
        const float inv_d = (d > 1e-6f) ? (1.f / d) : 0.f;
        Lgh_x = -best_bx * inv_d;
        Lgh_y = -best_by * inv_d;
        return true;
    }

    // ---- Filter tick ----

    void tick() {
        const auto now = this->now();
        auto stale = [&](const rclcpp::Time& t, double max_age_s) {
            if (t.nanoseconds() == 0) return true;
            return (now - t).seconds() > max_age_s;
        };

        // 1. No teleop -> publish zero.
        if (stale(last_teleop_time_, TELEOP_STALE_S)) {
            publish_safe(0.f, 0.f, 0.f, "NO_TELEOP");
            return;
        }

        // 2. NaN guard on inputs.
        if (!std::isfinite(teleop_vx_) || !std::isfinite(teleop_vy_)
            || !std::isfinite(teleop_wyaw_)
            || !std::isfinite(alpha_) || !std::isfinite(phi_)) {
            publish_safe(0.f, 0.f, 0.f, "NAN_INPUT");
            return;
        }

        // 3. Stale params -> passthrough (manual teleop still works while
        // inference is recovering; walking_bridge has its own watchdog).
        if (stale(last_params_time_, PARAMS_STALE_S)) {
            publish_safe(teleop_vx_, teleop_vy_, teleop_wyaw_, "PASSTHROUGH_PARAMS_STALE");
            return;
        }

        // 4. Stale LiDAR -> passthrough (don't block on perception dropout).
        if (stale(last_cloud_time_, LIDAR_STALE_S)) {
            publish_safe(teleop_vx_, teleop_vy_, teleop_wyaw_, "PASSTHROUGH_LIDAR_STALE");
            return;
        }

        float h, Lgh_x, Lgh_y;
        if (!compute_h_and_grad(h, Lgh_x, Lgh_y)) {
            publish_safe(teleop_vx_, teleop_vy_, teleop_wyaw_, "NO_OBSTACLE_DATA");
            return;
        }

        // 5. No obstacle in range -> passthrough.
        if (!std::isfinite(h)) {
            publish_safe(teleop_vx_, teleop_vy_, teleop_wyaw_, "OK_NO_OBSTACLE");
            publish_h(h, Lgh_x, Lgh_y, 0.f);
            return;
        }

        // 6. CBF QP — closed-form 2D projection.
        // Constraint:  L_g h . u  >=  -alpha * h  +  phi * ||L_g h||^2
        const float Lgh_norm_sq = Lgh_x * Lgh_x + Lgh_y * Lgh_y;
        const float rhs   = -alpha_ * h + phi_ * Lgh_norm_sq;
        const float dot   = Lgh_x * teleop_vx_ + Lgh_y * teleop_vy_;
        const float slack = dot - rhs;

        float u_safe_x = teleop_vx_;
        float u_safe_y = teleop_vy_;
        const char* state = "OK";

        if (slack < 0.f) {
            if (Lgh_norm_sq < 1e-8f) {
                // Constraint binding but no direction to move -> emergency stop.
                publish_safe(0.f, 0.f, 0.f, "INFEASIBLE");
                publish_h(h, Lgh_x, Lgh_y, slack);
                return;
            }
            const float proj_scale = slack / Lgh_norm_sq;
            u_safe_x = teleop_vx_ - Lgh_x * proj_scale;
            u_safe_y = teleop_vy_ - Lgh_y * proj_scale;
            state = "OK_DEFLECTED";
        }

        // Yaw stays unfiltered — CBF only constrains xy.
        publish_safe(u_safe_x, u_safe_y, teleop_wyaw_, state);
        publish_h(h, Lgh_x, Lgh_y, slack);
    }

    void publish_safe(float vx, float vy, float wyaw, const char* state) {
        if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(wyaw)) {
            vx = vy = wyaw = 0.f;
            state = "NAN_OUTPUT";
        }
        geometry_msgs::msg::Twist out;
        out.linear.x  = vx;
        out.linear.y  = vy;
        out.angular.z = wyaw;
        u_des_pub_->publish(out);
        last_state_ = state;
    }

    void publish_h(float h, float Lgh_x, float Lgh_y, float slack) {
        std_msgs::msg::Float32MultiArray msg;
        msg.data = {h, Lgh_x, Lgh_y, slack};
        h_pub_->publish(msg);
    }

    void heartbeat() {
        std_msgs::msg::String msg;
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "state=%s  alpha=%.2f  phi=%.2f  |cmd|=%.2f",
                      last_state_,
                      static_cast<double>(alpha_),
                      static_cast<double>(phi_),
                      std::sqrt(static_cast<double>(
                          teleop_vx_*teleop_vx_ + teleop_vy_*teleop_vy_)));
        msg.data = buf;
        status_pub_->publish(msg);
    }

    // ---- Members ----

    std::string cloud_topic_;
    std::string body_frame_;
    float safety_margin_;

    // Inputs.
    float teleop_vx_   = 0.f;
    float teleop_vy_   = 0.f;
    float teleop_wyaw_ = 0.f;
    rclcpp::Time last_teleop_time_{0, 0, RCL_ROS_TIME};

    // MVP action space: (alpha, phi). Hand-tuned defaults match
    // cbf_inference_node's SAFE_DEFAULTS.
    float alpha_ = 2.0f;
    float phi_   = 0.5f;
    rclcpp::Time last_params_time_{0, 0, RCL_ROS_TIME};

    sensor_msgs::msg::PointCloud2::SharedPtr latest_cloud_;
    rclcpp::Time last_cloud_time_{0, 0, RCL_ROS_TIME};

    const char* last_state_ = "INIT";

    // ROS handles.
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr teleop_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr params_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr u_des_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr h_pub_;
    rclcpp::TimerBase::SharedPtr tick_timer_;
    rclcpp::TimerBase::SharedPtr heartbeat_timer_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};


int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CbfFilterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
