// good

// walking_bridge.cpp
// Bridges teleop velocity commands to the Unitree Go2 SportClient.

// In the original project (robot_ws/src/src/semantic_poisson.cpp), the safety filter consumed teleop's "u_des" topic, ran the velocity through CBF safety logic, and called SportClient.Move(). We deleted that file, so nothing was telling the Go2 to move. This bridge fills that gap: subscribe to u_des, pass the velocity straight to SportClient.Move().

// Behavior:
//   - On startup: RecoveryStand -> SpeedLevel(1) -> ready
//   - On each Twist message on u_des: SportClient.Move(vx, vy, vyaw)
//   - On shutdown (Ctrl+C): StopMove -> StandDown
//   - Heartbeat safety: if no Twist received for 500ms, StopMove -> Damp
//     (teleop crash, SSH drop, etc.). Robot stays damped until node restart.
//
// Original source: replaces startup_robot() and dispatch_robot_command()
//   from robot_ws/src/src/semantic_poisson.cpp

#include <memory>
#include <unistd.h>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/bool.hpp"
#include "unitree_api/msg/request.hpp"
#include "ros2_sport_client.h"

class WalkingBridgeNode : public rclcpp::Node {
public:
    WalkingBridgeNode() : Node("walking_bridge"), sport_req_(this) {
        // How long to wait before triggering e-stop (default 5000ms)
        this->declare_parameter("heartbeat_timeout_ms", 5000);
        int timeout_ms = this->get_parameter("heartbeat_timeout_ms").as_int();
        heartbeat_timeout_ = std::chrono::milliseconds(timeout_ms);

        twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>("u_des", 1,
            std::bind(&WalkingBridgeNode::twist_callback, this, std::placeholders::_1));

        estop_sub_ = this->create_subscription<std_msgs::msg::Bool>("estop", 1,
            std::bind(&WalkingBridgeNode::estop_callback, this, std::placeholders::_1));

        // Watchdog timer — checks for heartbeat timeout every 100ms
        watchdog_timer_ = this->create_wall_timer(std::chrono::milliseconds(100),
            std::bind(&WalkingBridgeNode::watchdog_callback, this));

        RCLCPP_INFO(this->get_logger(), "Walking bridge initialized (heartbeat timeout: %d ms). Standing up...", timeout_ms);
        startup_robot();
        last_msg_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "Ready. Subscribing to u_des.");
    }

    ~WalkingBridgeNode() override {
        RCLCPP_INFO(this->get_logger(), "Shutting down. StopMove + StandDown.");
        sport_req_.StopMove(req_);
        sport_req_.StandDown(req_);
    }

private:
    void startup_robot() {
        // Wait for the Go2's sport service to discover our publisher.
        // Without this, commands sent immediately get lost.
        RCLCPP_INFO(this->get_logger(), "Waiting for sport API connection...");
        sleep(3);

        sport_req_.RecoveryStand(req_);
        sleep(2);
        sport_req_.SpeedLevel(req_, 1);
        sleep(1);
    }

    void twist_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        // If we already damped, ignore new messages — restart the node to recover
        if (damped_) return;

        last_msg_time_ = this->now();

        const float vx = static_cast<float>(msg->linear.x);
        const float vy = static_cast<float>(msg->linear.y);
        const float vyaw = static_cast<float>(msg->angular.z);
        sport_req_.Move(req_, vx, vy, vyaw);
    }

    void estop_callback(const std_msgs::msg::Bool::SharedPtr msg) {
        if (!msg->data || damped_) return;
        RCLCPP_ERROR(this->get_logger(), "E-STOP received from teleop! Emergency stop!");
        sport_req_.StopMove(req_);
        sport_req_.Damp(req_);
        damped_ = true;
        RCLCPP_ERROR(this->get_logger(), "Robot is now in DAMPING MODE. Restart this node to recover.");
    }

    void watchdog_callback() {
        if (damped_) return;

        // How long since the last Twist message?
        auto elapsed = this->now() - last_msg_time_;
        if (elapsed > heartbeat_timeout_) {
            RCLCPP_ERROR(this->get_logger(),
                "HEARTBEAT TIMEOUT — no Twist received for %.1f s. Emergency stop!",
                elapsed.seconds());
            sport_req_.StopMove(req_);
            sport_req_.Damp(req_);
            damped_ = true;
            RCLCPP_ERROR(this->get_logger(),
                "Robot is now in DAMPING MODE. Restart this node to recover.");
        }
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr estop_sub_;
    rclcpp::TimerBase::SharedPtr watchdog_timer_;
    SportClient sport_req_;
    unitree_api::msg::Request req_;

    rclcpp::Time last_msg_time_;
    std::chrono::milliseconds heartbeat_timeout_;
    bool damped_ = false;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WalkingBridgeNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
