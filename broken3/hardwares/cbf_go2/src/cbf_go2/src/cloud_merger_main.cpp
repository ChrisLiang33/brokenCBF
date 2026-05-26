//good

// cloud_merger_main.cpp
// Standalone CloudMerger node entry point.

// In the original project (robot_ws/src/src/semantic_poisson.cpp), CloudMergerNode
// was instantiated inside main() alongside PoissonControllerNode on a shared executor.
// Here it runs as its own standalone node since we don't need the Poisson pipeline.

#include "cloud_merger.h"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CloudMergerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
