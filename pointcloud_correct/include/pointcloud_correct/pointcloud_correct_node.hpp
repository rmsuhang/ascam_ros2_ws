#ifndef POINTCLOUD_CORRECT_NODE_HPP_
#define POINTCLOUD_CORRECT_NODE_HPP_

// ros2
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

//std
#include <memory>
#include <string>
#include <vector>

//pcl
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Geometry>
#include <chrono>

namespace pointcloud_correct
{
class PointcloudCorrectNode : public rclcpp::Node
{
public:
    PointcloudCorrectNode(const rclcpp::NodeOptions & options);  

private:    
    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void updateTransformMatrix();
    void transformPointCloudWithPCL(const sensor_msgs::msg::PointCloud2::SharedPtr input_cloud,
                                   sensor_msgs::msg::PointCloud2& output_cloud);
    void transformSinglePoint(const pcl::PointXYZ& input, pcl::PointXYZ& output);
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string target_frame_;
    std::string source_frame_;

    double timestamp_offset_ = 0;

    Eigen::Affine3f transform_matrix_;  // 预计算的变换矩阵
};
}  // namespace pointcloud_correct
#endif 