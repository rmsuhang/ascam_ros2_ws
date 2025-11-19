// std
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
// ros2
// ros2
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <Eigen/Geometry>

//project
#include "pointcloud_correct/pointcloud_correct_node.hpp"


namespace pointcloud_correct
{
PointcloudCorrectNode::PointcloudCorrectNode(const rclcpp::NodeOptions & options)
    : Node("pointcloud_correct_node", options)
{
    this->declare_parameter<std::string>("target_frame", "ascamera_hp60c_camera_link_0_corrected");
    this->declare_parameter<std::string>("source_frame", "ascamera_hp60c_camera_link_0");
    this->declare_parameter<double>("pitch_angle", 0.0); 
    this->declare_parameter<double>("roll_angle", 0.0); 
    this->declare_parameter<double>("yaw_angle", 0.0); 
    timestamp_offset_ = this->declare_parameter("timestamp_offset", 0.0);

    this->declare_parameter<bool>("use_pcl_transform", true);  // 新增参数：使用PCL变换
    this->declare_parameter<bool>("use_parallel_processing", true);  // 新增参数：使用并行处理

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
    tf_buffer_->setCreateTimerInterface(timer_interface);
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // 使用兼容的 QoS 配置
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/ascamera_hp60c/camera_publisher/depth0/points", 
        rclcpp::QoS(rclcpp::KeepLast(5)),  // 使用 KeepLast 而不是 SensorDataQoS
        std::bind(&PointcloudCorrectNode::pointcloudCallback, this, std::placeholders::_1));

    pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/ascamera_hp60c_corrected/camera_publisher/depth0/points", 
        rclcpp::QoS(rclcpp::KeepLast(5)));  // 使用相同的 QoS 配置

    updateTransformMatrix();// 预计算变换矩阵

    RCLCPP_INFO(this->get_logger(), "pointcloud_correct node initialize successfully");

}

void PointcloudCorrectNode::updateTransformMatrix()
{
    auto roll = this->get_parameter("roll_angle").as_double();
    auto pitch = this->get_parameter("pitch_angle").as_double();
    auto yaw = this->get_parameter("yaw_angle").as_double();
    
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    
    // 预计算变换矩阵
    transform_matrix_ = Eigen::Affine3f::Identity();
    transform_matrix_.rotate(Eigen::Quaternionf(q.w(), q.x(), q.y(), q.z()));

    RCLCPP_INFO(this->get_logger(), "Transform matrix updated");
}


void PointcloudCorrectNode::pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    geometry_msgs::msg::TransformStamped t;
    timestamp_offset_ = this->get_parameter("timestamp_offset").as_double();
    t.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset_);
    t.header.frame_id = this->get_parameter("source_frame").as_string();
    t.child_frame_id = this->get_parameter("target_frame").as_string();

    auto roll = this->get_parameter("roll_angle").as_double();
    auto pitch = this->get_parameter("pitch_angle").as_double();
    auto yaw = this->get_parameter("yaw_angle").as_double();
    tf2::Quaternion q;
    // std::cout<<"r: "<<roll<<"p: "<<pitch<<"y: "<<yaw<<std::endl;
    q.setRPY(roll, pitch, yaw);

    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();

    t.transform.translation.x = 0.0;
    t.transform.translation.y = 0.0;
    t.transform.translation.z = 0.0;

    tf_broadcaster_->sendTransform(t);

    sensor_msgs::msg::PointCloud2 transformed_cloud;
   
    bool use_pcl_transform = this->get_parameter("use_pcl_transform").as_bool();

    if (use_pcl_transform) {
        //使用pcl变换
        transformPointCloudWithPCL(msg, transformed_cloud);
    } else {
        // 使用tf2变换
        try {
            tf2::doTransform(*msg, transformed_cloud, t);
        } catch (const tf2::TransformException & ex) {
            RCLCPP_ERROR(this->get_logger(), "Transform failed: %s", ex.what());
            pointcloud_pub_->publish(*msg);
            return;
        }
    }

    //发布变换后的点云
    pointcloud_pub_->publish(transformed_cloud);
    
}

void PointcloudCorrectNode::transformPointCloudWithPCL(const sensor_msgs::msg::PointCloud2::SharedPtr input_cloud,
                                                      sensor_msgs::msg::PointCloud2& output_cloud)
{
    // 将ROS消息转换为PCL点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input_cloud, *pcl_cloud);
    
    // 创建输出点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    transformed_cloud->header = pcl_cloud->header;
    transformed_cloud->is_dense = pcl_cloud->is_dense;
    transformed_cloud->width = pcl_cloud->width;
    transformed_cloud->height = pcl_cloud->height;
    transformed_cloud->points.resize(pcl_cloud->points.size());
    
    bool use_parallel = this->get_parameter("use_parallel_processing").as_bool();
    
    if (use_parallel) {
        // 使用OpenMP并行处理（如果可用）
        #pragma omp parallel for
        for (size_t i = 0; i < pcl_cloud->points.size(); ++i) {
            transformSinglePoint(pcl_cloud->points[i], transformed_cloud->points[i]);
        }
    } else {
        // 串行处理
        for (size_t i = 0; i < pcl_cloud->points.size(); ++i) {
            transformSinglePoint(pcl_cloud->points[i], transformed_cloud->points[i]);
        }
    }
    
    // 转换回ROS消息
    pcl::toROSMsg(*transformed_cloud, output_cloud);
    output_cloud.header = input_cloud->header;
    output_cloud.header.frame_id = this->get_parameter("source_frame").as_string();//在原始坐标系下发布变换后的点云，修正后的坐标系是假想出来的，是为了完成坐标变换
}

void PointcloudCorrectNode::transformSinglePoint(const pcl::PointXYZ& input, pcl::PointXYZ& output)
{
    // 跳过无效点
    if (!std::isfinite(input.x) || !std::isfinite(input.y) || !std::isfinite(input.z)) {
        output = input;
        return;
    }
    
    // 使用预计算的变换矩阵
    Eigen::Vector3f point(input.x, input.y, input.z);
    Eigen::Vector3f transformed_point = transform_matrix_ * point;
    
    output.x = transformed_point.x();
    output.y = transformed_point.y();
    output.z = transformed_point.z();
}

}  // namespace pointcloud_correct

#include "rclcpp_components/register_node_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_correct::PointcloudCorrectNode)