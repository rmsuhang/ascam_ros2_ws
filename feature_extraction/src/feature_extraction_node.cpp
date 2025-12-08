// std
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <numeric>  

// ROS2
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/convex_hull.h>

// Eigen
#include <Eigen/Core>

// Project
#include "feature_extraction/feature_extraction_node.hpp"

namespace feature_extraction
{

FeatureExtractionNode::FeatureExtractionNode(const rclcpp::NodeOptions & options)
    : Node("feature_extraction_node", options),
      processed_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
      height_cloud_(new pcl::PointCloud<pcl::PointXYZI>),
      processed_count_(0)
{
    // 参数声明
    this->declare_parameter<double>("reference_plane_distance", 0.5);  // 参考平面到相机的距离（米）
    this->declare_parameter<double>("feature_threshold", 0.002);       // 特征检测阈值（米）
    this->declare_parameter<double>("min_feature_height", 0.003);      // 最小特征高度（米）
    this->declare_parameter<double>("max_feature_height", 0.01);       // 最大特征高度（米）
    this->declare_parameter<double>("voxel_leaf_size", 0.005);         // 体素滤波叶子大小
    this->declare_parameter<int>("statistical_mean_k", 20);           // 统计滤波邻域点数
    this->declare_parameter<double>("statistical_stddev_mult", 1.0);  // 统计滤波标准差倍数
    this->declare_parameter<double>("top_percentage", 20.0);          // 统计前百分之多少的高度差
    this->declare_parameter<double>("min_area_points", 100.0);         // 计算面积所需的最小点数
    
    // 获取参数
    reference_plane_distance_ = this->get_parameter("reference_plane_distance").as_double();
    feature_threshold_ = this->get_parameter("feature_threshold").as_double();
    min_feature_height_ = this->get_parameter("min_feature_height").as_double();
    max_feature_height_ = this->get_parameter("max_feature_height").as_double();
    voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();// 体素滤波叶子大小，它通过将点云空间划分为规则的体素网格，
                                                                          //然后用每个体素内所有点的重心或中心点来代表该体素内的点
                                                                          //从而减少点云数量，同时保持点云的宏观形状特征。
    statistical_mean_k_ = this->get_parameter("statistical_mean_k").as_int();
    statistical_stddev_mult_ = this->get_parameter("statistical_stddev_mult").as_double();
    top_percentage_ = this->get_parameter("top_percentage").as_double();
    min_area_points_ = this->get_parameter("min_area_points").as_double();

    // 订阅校正后的点云
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/ascamera_hp60c_corrected/camera_publisher/depth0/points", 
        rclcpp::QoS(rclcpp::KeepLast(10)),
        std::bind(&FeatureExtractionNode::pointcloudCallback, this, std::placeholders::_1));
    
    // 发布特征点云
    feature_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/features/extracted_features", 
        rclcpp::QoS(rclcpp::KeepLast(10)));
    
    // 发布高度图点云
    height_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/features/height_map", 
        rclcpp::QoS(rclcpp::KeepLast(10)));
    
    // 发布显著特征可视化标记
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/features/visualization_markers", 
        rclcpp::QoS(rclcpp::KeepLast(10)));

    //高度差统计数据 
    height_stats_pub_ = this->create_publisher<std_msgs::msg::Float32>(
        "/features/height_statistics", 
        rclcpp::QoS(rclcpp::KeepLast(10)));

    // 发布边界框尺寸数据
    bbox_size_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
        "/features/bounding_box_dimensions", 
        rclcpp::QoS(rclcpp::KeepLast(10)));

    RCLCPP_INFO(this->get_logger(), 
                "Feature extraction node initialized with reference plane distance: %.3f m", 
                reference_plane_distance_);

    RCLCPP_INFO(this->get_logger(), 
                "Top percentage for height statistics: %.1f%%", top_percentage_);

    RCLCPP_INFO(this->get_logger(), 
                "Minimum points for area calculation: %.0f", min_area_points_);
}

void FeatureExtractionNode::pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    //auto start_time = std::chrono::high_resolution_clock::now();
    
    
    // 提取特征
    extractFeatures(msg);
    
    // 发布高度图
    publishHeightMap(msg);
    
    processed_count_++;
    
    // // 定期输出统计信息
    // if (processed_count_ % 100 == 0) {
    //     auto end_time = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    //     RCLCPP_DEBUG(this->get_logger(), "Feature extraction time: %ld ms", duration.count());
    // }
}

void FeatureExtractionNode::extractFeatures(const sensor_msgs::msg::PointCloud2::SharedPtr input_cloud_msg)
{

    // 转换为PCL点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*input_cloud_msg, *pcl_cloud_);
    
    // 1. 预处理：体素滤波降采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(pcl_cloud_);
    voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel_filter.filter(*filtered_cloud);
    
    // 2. 统计滤波去除离群点
    pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor_filter;
    sor_filter.setInputCloud(filtered_cloud);//设置输入点云
    sor_filter.setMeanK(statistical_mean_k_);//设置查询点的邻域点数
    sor_filter.setStddevMulThresh(statistical_stddev_mult_);//设置标准差倍数阈值
    sor_filter.filter(*inlier_cloud);//执行滤波，保存内点到inlier_cloud
    
    // 3. 计算高度差并提取特征点
    pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<pcl::PointXYZ> significant_features;
    std::vector<double> height_differences;  // 存储所有特征点的高度差

    for (const auto& point : inlier_cloud->points) {

        //RCLCPP_INFO(this->get_logger(), "point.z : %.3f m", point.z);
        
        // 计算相对于参考平面的高度
        double height_difference = reference_plane_distance_- point.z ;
        
        // 检查是否超过特征阈值
        if (height_difference> feature_threshold_ && 
            height_difference >= min_feature_height_ && 
            height_difference <= max_feature_height_) {
            
            pcl::PointXYZI feature_point;
            feature_point.x = point.x;
            feature_point.y = point.y;
            feature_point.z = point.z;
            feature_point.intensity = height_difference;  // 强度值存储高度差
            
            feature_cloud->points.push_back(feature_point);
            height_differences.push_back(height_difference);  // 储存高度差

            // 记录显著特征点用于可视化
            if (height_difference > feature_threshold_ * 2) {
                significant_features.push_back(point);
            }
        }
    }

    // 4. 计算特征点的边界框尺寸
    auto bbox_dimensions = calculateBoundingBoxDimensionsOptimized(feature_cloud);

    // 5. 发布特征点云
    if (!feature_cloud->empty()) {
        sensor_msgs::msg::PointCloud2 feature_msg;
        pcl::toROSMsg(*feature_cloud, feature_msg);
        feature_msg.header = input_cloud_msg->header;//与输入点云的信息同步
        feature_msg.header.frame_id = input_cloud_msg->header.frame_id;
        feature_pub_->publish(feature_msg);
        
        // 发布可视化标记
        //publishFeatureMarkers(significant_features);
        
        RCLCPP_DEBUG(this->get_logger(), "Extracted %zu feature points", feature_cloud->size());
        
        // 发布边界框尺寸数据
        geometry_msgs::msg::Vector3 bbox_msg;
        bbox_msg.x = bbox_dimensions.first;  // 长度
        bbox_msg.y = bbox_dimensions.second; // 宽度
        bbox_msg.z = bbox_dimensions.first * bbox_dimensions.second; // 面积（可选）
        bbox_size_pub_->publish(bbox_msg);

        RCLCPP_INFO(this->get_logger(), 
            "Bounding box dimensions - Length: %.6f m, Width: %.6f m, Area: %.6f m² (based on %zu points)", 
            bbox_dimensions.first, bbox_dimensions.second, 
            bbox_dimensions.first * bbox_dimensions.second, feature_cloud->size());
    }
    

    // 6. 统计高度差并发布结果
    if (!height_differences.empty()) {
        double top_height_average = calculateTopHeightAverage(height_differences);
        
        // 发布高度差统计数据
        std_msgs::msg::Float32 height_stats_msg;
        height_stats_msg.data = top_height_average;
        height_stats_pub_->publish(height_stats_msg);
        
        // RCLCPP_INFO(this->get_logger(), 
        //            "Height statistics - Total features: %zu, Top %.1f%% average height: %.4f m", 
        //            height_differences.size(), top_percentage_, top_height_average);
    }
}

std::pair<double, double> FeatureExtractionNode::calculateBoundingBoxDimensionsOptimized(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud)
{
    if (feature_cloud->size() < min_area_points_) {
        return std::make_pair(0.0, 0.0);
    }
    
    try {
        // 1. 复制点云并按X坐标排序
        std::vector<pcl::PointXYZI> sorted_points(feature_cloud->begin(), feature_cloud->end());
        std::sort(sorted_points.begin(), sorted_points.end(),
            [](const pcl::PointXYZI& a, const pcl::PointXYZI& b) {
                return a.x < b.x;  // Y从小到大排序
            });
        
        // 2. 确定X轴范围
        float x_min = sorted_points.front().x;
        float x_max = sorted_points.back().x;
        float x_range = x_max - x_min;
        
        if (x_range < 1e-6) {
            return std::make_pair(0.0, 0.0);
        }
        
        // 3. 将X范围等分成N个区间
        const int N = 10;
        double max_width = 0.0;
        Eigen::Vector2f left_pt, right_pt;
        
        // 预计算每个区间的Y边界
        std::vector<float> slice_boundaries(N + 1);
        for (int i = 0; i <= N; ++i) {
            slice_boundaries[i] = x_min + i * x_range / N;
        }
        
        // 4. 遍历每个区间
        for (int i = 0; i < N; ++i) {
            float slice_x_min = slice_boundaries[i];
            float slice_x_max = slice_boundaries[i + 1];
            
            // 使用二分查找快速找到区间内的点
            auto lower = std::lower_bound(sorted_points.begin(), sorted_points.end(), 
                slice_x_min, [](const pcl::PointXYZI& p, float x) { return p.x < x; });
            auto upper = std::upper_bound(sorted_points.begin(), sorted_points.end(), 
                slice_x_max, [](float x, const pcl::PointXYZI& p) { return x < p.x; });
            
            // 如果区间内有点，计算Y范围
            if (lower != upper) {
                float min_y = std::numeric_limits<float>::max();
                float max_y = -std::numeric_limits<float>::max();
                Eigen::Vector2f current_left_pt, current_right_pt;
                
                for (auto it = lower; it != upper; ++it) {
                    if (it->y < min_y) {
                        min_y = it->y;
                        current_left_pt = Eigen::Vector2f(it->x, it->y);
                    }
                    if (it->y > max_y) {
                        max_y = it->y;
                        current_right_pt = Eigen::Vector2f(it->x, it->y);
                    }
                }
                
                float width = max_y - min_y;
                if (width > max_width) {
                    max_width = width;
                    left_pt = current_left_pt;
                    right_pt = current_right_pt;
                }
            }
        }
        
        if (max_width < 1e-6) {
            return std::make_pair(0.0, 0.0);
        }
        
        // 5. 寻找最下面的点（已经排序，最后一个点就是X最大的）
        Eigen::Vector2f bottom_point(sorted_points.back().x, sorted_points.back().y);
        
        // 6. 计算短直径
        float short_radius = calculatePointToLineDistance(bottom_point, left_pt, right_pt);
        double short_diameter = short_radius * 2.0;
        
        return std::make_pair(max_width, short_diameter);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error: %s", e.what());
        return std::make_pair(0.0, 0.0);
    }
}


// 辅助函数：计算点到直线的距离（返回绝对值）
float FeatureExtractionNode::calculatePointToLineDistance(
    const Eigen::Vector2f& point, 
    const Eigen::Vector2f& line_start, 
    const Eigen::Vector2f& line_end)
{
    // 向量AB
    Eigen::Vector2f ab = line_end - line_start;
    // 向量AP
    Eigen::Vector2f ap = point - line_start;
    
    // 计算投影长度
    float proj_length = ap.dot(ab) / ab.dot(ab);
    
    // 计算垂足坐标
    Eigen::Vector2f projection;
    if (proj_length <= 0.0f) {
        projection = line_start;
    } else if (proj_length >= 1.0f) {
        projection = line_end;
    } else {
        projection = line_start + proj_length * ab;
    }
    
    // 返回点到垂足的距离（绝对值）
    float distance = (point - projection).norm();
    
    // 确保距离为正数
    return std::abs(distance);
}

std::pair<double, double> FeatureExtractionNode::calculateBoundingBoxDimensions(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud)
{
    if (feature_cloud->size() < min_area_points_) {
        RCLCPP_DEBUG(this->get_logger(), 
                    "Not enough points for bounding box calculation: %zu < %.0f", 
                    feature_cloud->size(), min_area_points_);
        return std::make_pair(0.0, 0.0);
    }
    
    // 计算边界框
    pcl::PointXYZI min_pt, max_pt;
    pcl::getMinMax3D<pcl::PointXYZI>(*feature_cloud, min_pt, max_pt);
    
    double length = max_pt.x - min_pt.x;   // X方向尺寸
    double width = max_pt.y - min_pt.y;    // Y方向尺寸
    
    RCLCPP_DEBUG(this->get_logger(), 
                "Bounding box - Length (X): %.6f m, Width (Y): %.6f m", 
                length, width);
    
    return std::make_pair(length, width);
}


// double FeatureExtractionNode::calculateBoundingBoxArea(const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud)
// {
//     // 方法2: 计算边界框面积（简单但不够准确）
//     pcl::PointXYZI min_pt, max_pt;
    
//     // 使用正确的模板特化
//     pcl::getMinMax3D<pcl::PointXYZI>(*feature_cloud, min_pt, max_pt);
    
//     double width = max_pt.x - min_pt.x;
//     double height = max_pt.y - min_pt.y;
//     double bbox_area = width * height;
    
//     RCLCPP_DEBUG(this->get_logger(), 
//                 "Bounding box area: %.6f m² (%.3f x %.3f m)", 
//                 bbox_area, width, height);
    
//     return bbox_area;
// }

double FeatureExtractionNode::calculateTopHeightAverage(const std::vector<double>& height_differences)
{
    if (height_differences.empty()) {
        return 0.0;
    }
    
    // 复制高度差数据以便排序
    std::vector<double> sorted_heights = height_differences;
    
    // 从大到小排序
    std::sort(sorted_heights.begin(), sorted_heights.end(), std::greater<double>());
    
    // 计算前百分之多少的点数
    size_t top_count = static_cast<size_t>(sorted_heights.size() * (top_percentage_ / 100.0));
    
    // 确保至少有一个点
    if (top_count == 0) {
        top_count = 1;
    }
    
    // 确保不超过总数
    if (top_count > sorted_heights.size()) {
        top_count = sorted_heights.size();
    }
    
    // 计算前top_count个点的平均值
    double sum = 0.0;
    for (size_t i = 0; i < top_count; ++i) {
        sum += sorted_heights[i];
    }
    
    double average = sum / top_count;
    
    // 输出详细统计信息
    RCLCPP_DEBUG(this->get_logger(), 
                "Height statistics details: Total=%zu, Top %zu points (%.1f%%), "
                "Min=%.4f, Max=%.4f, Average=%.4f m", 
                sorted_heights.size(), top_count, top_percentage_,
                sorted_heights.back(), sorted_heights.front(), average);
    
    return average;
}

void FeatureExtractionNode::publishHeightMap(const sensor_msgs::msg::PointCloud2::SharedPtr input_cloud_msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input_cloud_msg, *pcl_cloud_);

    // 创建高度图点云
    height_cloud_->clear();
    height_cloud_->header = pcl_cloud_->header;
    height_cloud_->width = pcl_cloud_->width;
    height_cloud_->height = pcl_cloud_->height;
    height_cloud_->is_dense = pcl_cloud_->is_dense;
    height_cloud_->points.resize(pcl_cloud_->points.size());
    
    // 计算每个点的高度差
    for (size_t i = 0; i < pcl_cloud_->points.size(); ++i) {
        const auto& input_point = pcl_cloud_->points[i];
        auto& output_point = height_cloud_->points[i];
        
        output_point.x = input_point.x;
        output_point.y = input_point.y;
        output_point.z = input_point.z;
        
        // 强度值表示相对于参考平面的高度差
        double height_difference = input_point.z - reference_plane_distance_;
        output_point.intensity = height_difference;
    }
    
    // 发布高度图
    sensor_msgs::msg::PointCloud2 height_msg;
    pcl::toROSMsg(*height_cloud_, height_msg);
    height_msg.header = input_cloud_msg->header;
    height_msg.header.frame_id = input_cloud_msg->header.frame_id;
    height_map_pub_->publish(height_msg);
}

void FeatureExtractionNode::publishFeatureMarkers(const std::vector<pcl::PointXYZ>& features)
{
    visualization_msgs::msg::MarkerArray marker_array;
    
    // 清除之前的标记
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "ascamera_hp60c_camera_link_0_corrected";
    clear_marker.header.stamp = this->now();
    clear_marker.ns = "features";
    clear_marker.id = 0;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);
    
    // 创建特征点标记
    for (size_t i = 0; i < features.size(); ++i) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "ascamera_hp60c_camera_link_0_corrected";
        marker.header.stamp = this->now();
        marker.ns = "features";
        marker.id = i + 1;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // 设置位置
        marker.pose.position.x = features[i].x;
        marker.pose.position.y = features[i].y;
        marker.pose.position.z = features[i].z;
        marker.pose.orientation.w = 1.0;
        
        // 设置大小和颜色
        marker.scale.x = 0.05;  // 5cm 半径
        marker.scale.y = 0.05;
        marker.scale.z = 0.05;
        
        marker.color.r = 1.0;  // 红色
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;  // 半透明
        
        marker.lifetime = rclcpp::Duration::from_seconds(1.0);  // 1秒生命周期
        
        marker_array.markers.push_back(marker);
    }
    
    // 发布标记
    marker_pub_->publish(marker_array);
}

} // namespace feature_extraction

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(feature_extraction::FeatureExtractionNode)