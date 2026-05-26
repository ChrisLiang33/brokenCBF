//good

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <ncurses.h>

class TeleopNode : public rclcpp::Node {
    public:
        TeleopNode() : Node("teleop_node"){
            // Velocity bound parameters
            this->declare_parameter("vel_max_x_fwd", 0.9);
            this->declare_parameter("vel_max_x_bwd", 0.9);
            this->declare_parameter("vel_max_y", 0.9);
            this->declare_parameter("vel_max_yaw", 0.8);
            vel_max_x_fwd_ = this->get_parameter("vel_max_x_fwd").as_double();
            vel_max_x_bwd_ = this->get_parameter("vel_max_x_bwd").as_double();
            vel_max_y_ = this->get_parameter("vel_max_y").as_double();
            vel_max_yaw_ = this->get_parameter("vel_max_yaw").as_double();
            RCLCPP_INFO(this->get_logger(), "Velocity bounds: x_fwd=%.2f, x_bwd=%.2f, y=%.2f,       yaw=%.2f", vel_max_x_fwd_, vel_max_x_bwd_, vel_max_y_, vel_max_yaw_);

            // V14 (2026-05-21): teleop now publishes the RAW user command on
            // /u_teleop. The CBF filter node sits between teleop and the
            // walking_bridge: it consumes /u_teleop + /cbf/params + /cbf/grid,
            // solves the safety QP, and republishes the filtered output on
            // /u_des — which walking_bridge still consumes unchanged.
            twist_puber_ = this->create_publisher<geometry_msgs::msg::Twist>("u_teleop", 1);
            key_puber_ = this->create_publisher<std_msgs::msg::Int32>("key_press", 1);
            estop_puber_ = this->create_publisher<std_msgs::msg::Bool>("estop", 1);
            teleop_timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&TeleopNode::teleop_callback, this));

            initscr();
            cbreak();
            noecho();
            nodelay(stdscr, TRUE);
            keypad(stdscr, TRUE);
            printw("Use Arrow Keys to Move, SPACE = E-STOP, 'q' to quit.\n");
            refresh();
        }

        ~TeleopNode() override {
            endwin();
        }

    private:
        void teleop_callback(void){
            int ch = getch();
            switch(ch){
                case KEY_UP:
                    vxb = vel_max_x_fwd_;
                    break;
                case KEY_DOWN:
                    vxb = -vel_max_x_bwd_;
                    break;
                case KEY_LEFT:
                    vyb = vel_max_y_;
                    break;
                case KEY_RIGHT:
                    vyb = -vel_max_y_;
                    break;
                case ',':
                    vyaw = vel_max_yaw_;
                    break;
                case '.':
                    vyaw = -vel_max_yaw_;
                    break;
                case ' ':  // spacebar = e-stop
                {
                    std_msgs::msg::Bool estop_msg;
                    estop_msg.data = true;
                    estop_puber_->publish(estop_msg);
                    vxb = 0.0f;
                    vyb = 0.0f;
                    vyaw = 0.0f;
                    printw("\n*** E-STOP SENT ***\n");
                    refresh();
                    break;
                }
                case 'q':
                    rclcpp::shutdown();
                    return;
                case ERR:
                    break;
                default:
                    break;
            }

            if(ch==ERR){
                idle_counter++;
                if(idle_counter>20){
                    vxb = 0.0f;
                    vyb = 0.0f;
                    vyaw = 0.0f;
                }
            }
            else{
                idle_counter = 0;
            }

            geometry_msgs::msg::Twist twist_msg;
            twist_msg.linear.x = vxb;
            twist_msg.linear.y = vyb;
            twist_msg.linear.z = 0.0f;
            twist_msg.angular.x = 0.0f;
            twist_msg.angular.y = 0.0f;
            twist_msg.angular.z = vyaw;
            twist_puber_->publish(twist_msg);

            std_msgs::msg::Int32 key_msg;
            key_msg.data = ch;
            key_puber_->publish(key_msg);

        }

        volatile float vxb = 0.0f;
        volatile float vyb = 0.0f;
        volatile float vyaw = 0.0f;
        volatile int idle_counter = 0;
        
        // Velocity bounds
        float vel_max_x_fwd_ = 0.9f;
        float vel_max_x_bwd_ = 0.9f;
        float vel_max_y_ = 0.9f;
        float vel_max_yaw_ = 0.8f;

        rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr key_puber_;
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_puber_;
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr estop_puber_;
        rclcpp::TimerBase::SharedPtr teleop_timer_;

};

int main(int argc, char *argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TeleopNode>());
    rclcpp::shutdown();
    return 0;
}             
