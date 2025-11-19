#include "calibrate_point_cloud/calibrate_point_cloud.h"

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FilteringNode>("calibrate_node");
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}