#ifndef FEATURE_EXTRACTION_NODE_HPP
#define FEATURE_EXTRACTION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp> 
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/float32.hpp>

namespace feature_extraction
{

class FeatureExtractionNode : public rclcpp::Node
{
public:
    explicit FeatureExtractionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    
private:
    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void extractFeatures(const sensor_msgs::msg::PointCloud2::SharedPtr input_cloud_msg);
    void publishFeatureMarkers(const std::vector<pcl::PointXYZ>& features);
    void publishHeightMap(const sensor_msgs::msg::PointCloud2::SharedPtr input_cloud_msg);
    double calculateTopHeightAverage(const std::vector<double>& height_differences); 
    double calculateFeatureArea(const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud);
    double calculateBoundingBoxArea(const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud);

    std::pair<double, double> calculateBoundingBoxDimensions(const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud);
    std::pair<double, double> calculateBoundingBoxDimensionsOptimized(const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud);//计算两个方向的直径
    // 辅助函数：计算点到直线的距离（返回绝对值）
    float calculatePointToLineDistance(const Eigen::Vector2f& point, const Eigen::Vector2f& line_start, const Eigen::Vector2f& line_end);
    void publishROIMarker();


    void applyROIFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                                                pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud);
    

    // ROS2
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr feature_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr height_map_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr height_stats_pub_; 
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr bbox_size_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr roi_marker_pub_;
    
    // 参数
    double reference_plane_distance_;
    double feature_threshold_;
    double min_feature_height_;
    double max_feature_height_;
    double voxel_leaf_size_;
    int statistical_mean_k_;
    double statistical_stddev_mult_;
    double top_percentage_;  
    double min_area_points_;

    // ROI参数
    bool use_roi_;
    double roi_length_ratio_;
    double roi_width_ratio_;
    double roi_center_x_;
    double roi_center_y_;
    double roi_max_range_x_;
    double roi_max_range_y_;

    double roi_min_x_ ;
    double roi_max_x_ ;
    double roi_min_y_ ;
    double roi_max_y_ ;
    double roi_center_z_;
    
    
    // 预分配对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr processed_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr height_cloud_;
    
    // 统计信息
    size_t processed_count_;
};

} // namespace feature_extraction

#endif // FEATURE_EXTRACTION_NODE_HPP