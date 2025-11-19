#include "process_point_cloud/process_node.h"


int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node{std::make_shared<ProcessPointCloudNode>("process_point_cloud_node")};
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
