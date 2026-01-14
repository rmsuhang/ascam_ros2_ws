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

// Visualization
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// CvBridge
#include <cv_bridge/cv_bridge.h>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/crop_box.h>  // 添加CropBox滤波器
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/convex_hull.h>

// OpenCV
#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc.hpp>

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

    //图像处理参数
    this->declare_parameter<float>("image_resolution", 0.01f);
    this->declare_parameter<int>("morphology_kernel_size", 5);
    this->declare_parameter<int>("dilation_iterations", 1);
    this->declare_parameter<int>("erosion_iterations", 1);
    this->declare_parameter<bool>("use_dilation_first", true);
    this->declare_parameter<bool>("debug_display_image", true);
    
    // ROI区域参数
    this->declare_parameter<bool>("use_roi", true);                   // 是否使用ROI区域
    this->declare_parameter<double>("roi_length_ratio", 0.75);        // 长度方向ROI比例 (3/4)
    this->declare_parameter<double>("roi_width_ratio", 0.75);         // 宽度方向ROI比例 (3/4)
    this->declare_parameter<double>("roi_center_x", 0.0);             // ROI中心X坐标
    this->declare_parameter<double>("roi_center_y", 0.0);             // ROI中心Y坐标
    this->declare_parameter<double>("roi_max_range_x", 1.0);          // ROI最大X范围
    this->declare_parameter<double>("roi_max_range_y", 1.0);          // ROI最大Y范围
    
    // 获取参数
    reference_plane_distance_ = this->get_parameter("reference_plane_distance").as_double();
    feature_threshold_ = this->get_parameter("feature_threshold").as_double();
    min_feature_height_ = this->get_parameter("min_feature_height").as_double();
    max_feature_height_ = this->get_parameter("max_feature_height").as_double();
    voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
    statistical_mean_k_ = this->get_parameter("statistical_mean_k").as_int();
    statistical_stddev_mult_ = this->get_parameter("statistical_stddev_mult").as_double();
    top_percentage_ = this->get_parameter("top_percentage").as_double();
    min_area_points_ = this->get_parameter("min_area_points").as_double();
    
    // 获取图像处理参数
    image_resolution_ = this->get_parameter("image_resolution").as_double();
    morphology_kernel_size_ = this->get_parameter("morphology_kernel_size").as_int();
    dilation_iterations_ = this->get_parameter("dilation_iterations").as_int();
    erosion_iterations_ = this->get_parameter("erosion_iterations").as_int();
    use_dilation_first_ = this->get_parameter("use_dilation_first").as_bool();
    debug_display_image_ = this->get_parameter("debug_display_image").as_bool();

    // 获取ROI参数
    use_roi_ = this->get_parameter("use_roi").as_bool();
    roi_length_ratio_ = this->get_parameter("roi_length_ratio").as_double();
    roi_width_ratio_ = this->get_parameter("roi_width_ratio").as_double();
    roi_center_x_ = this->get_parameter("roi_center_x").as_double();
    roi_center_y_ = this->get_parameter("roi_center_y").as_double();
    roi_max_range_x_ = this->get_parameter("roi_max_range_x").as_double();
    roi_max_range_y_ = this->get_parameter("roi_max_range_y").as_double();

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

    // 高度差统计数据 
    height_stats_pub_ = this->create_publisher<std_msgs::msg::Float32>(
        "/features/height_statistics", 
        rclcpp::QoS(rclcpp::KeepLast(10)));

    // 发布边界框尺寸数据
    bbox_size_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
        "/features/bounding_box_dimensions", 
        rclcpp::QoS(rclcpp::KeepLast(10)));

    // 发布ROI区域可视化标记
    roi_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/features/roi_marker", 
        rclcpp::QoS(rclcpp::KeepLast(10)));

    // 二值图像发布器
    binary_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/features/binary_image", 
        rclcpp::QoS(rclcpp::KeepLast(10)));

    //xoy
    cloud_xoy_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/features/cloud_xoy", 
        rclcpp::QoS(rclcpp::KeepLast(10)));

    // 创建发布器
    visualization_pub_1 = this->create_publisher<sensor_msgs::msg::Image>(
        "/features/process_image", 10);

    // 创建发布器
    visualization_pub_2 = this->create_publisher<sensor_msgs::msg::Image>(
        "/features/contour_image", 10);


    RCLCPP_INFO(this->get_logger(), 
                "Feature extraction node initialized with reference plane distance: %.3f m", 
                reference_plane_distance_);

    RCLCPP_INFO(this->get_logger(), 
                "ROI enabled: %s, Length ratio: %.2f, Width ratio: %.2f", 
                use_roi_ ? "true" : "false", roi_length_ratio_, roi_width_ratio_);

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
    
    //processed_count_++;
    
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
    
    // 0. 预处理：使用ROI区域裁剪点云（可选步骤）
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (use_roi_) {
        applyROIFilter(pcl_cloud_, cropped_cloud);
        //RCLCPP_INFO(this->get_logger(), "ROI filtering: %zu -> %zu points", 
                     //pcl_cloud_->size(), cropped_cloud->size());
    } else {
        *cropped_cloud = *pcl_cloud_;
    }
    
    // 1. 预处理：体素滤波降采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cropped_cloud);
    voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel_filter.filter(*filtered_cloud);
    
    // 2. 统计滤波去除离群点
    pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor_filter;
    sor_filter.setInputCloud(filtered_cloud);
    sor_filter.setMeanK(statistical_mean_k_);
    sor_filter.setStddevMulThresh(statistical_stddev_mult_);
    sor_filter.filter(*inlier_cloud);

    
    // 3. 计算高度差并提取特征点
    pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<pcl::PointXYZ> significant_features;
    std::vector<double> height_differences;  // 存储所有特征点的高度差

    for (const auto& point : inlier_cloud->points) {
        // 计算相对于参考平面的高度
        double height_difference = std::abs(reference_plane_distance_ - point.z);
        
        // 检查是否超过特征阈值
        if (height_difference > feature_threshold_ && 
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
    //auto bbox_dimensions = calculateBoundingBoxDimensionsOptimized(feature_cloud);
    auto bbox_dimensions = calculateBoundingBoxWithMorphology(feature_cloud);

    // 5. 发布特征点云
    if (!feature_cloud->empty()) {
        sensor_msgs::msg::PointCloud2 feature_msg;
        pcl::toROSMsg(*feature_cloud, feature_msg);
        feature_msg.header = input_cloud_msg->header;
        feature_msg.header.frame_id = input_cloud_msg->header.frame_id;
        feature_pub_->publish(feature_msg);
        
        // 发布可视化标记
        //publishFeatureMarkers(significant_features);
        
        // 发布ROI区域标记
        if (use_roi_) {
            publishROIMarker();
        }
        
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
    }
}

// 应用ROI区域滤波
void FeatureExtractionNode::applyROIFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud)
{
    // 如果输入点云为空，直接返回
    if (input_cloud->empty()) {
        *output_cloud = *input_cloud;
        return;
    }
    
    try {
        // 计算点云的边界
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*input_cloud, min_pt, max_pt);
        
        // 计算点云的尺寸
        double cloud_length = max_pt.x - min_pt.x;
        double cloud_width = max_pt.y - min_pt.y;
        
        // 使用参数或自动计算ROI中心
        double roi_center_x = roi_center_x_;
        double roi_center_y = roi_center_y_;
        
        // 如果未指定中心，使用点云中心
        if (std::abs(roi_center_x) < 0.001 && std::abs(roi_center_y) < 0.001) {
            roi_center_x = (min_pt.x + max_pt.x) / 2.0;
            roi_center_y = (min_pt.y + max_pt.y) / 2.0;
        }
        
        // 计算ROI范围
        double roi_half_length = (cloud_length * roi_length_ratio_) / 2.0;
        double roi_half_width = (cloud_width * roi_width_ratio_) / 2.0;
        
        // // 限制最大范围
        // if (roi_half_length > roi_max_range_x_ / 2.0) {
        //     roi_half_length = roi_max_range_x_ / 2.0;
        // }
        // if (roi_half_width > roi_max_range_y_ / 2.0) {
        //     roi_half_width = roi_max_range_y_ / 2.0;
        // }
        
        // 计算ROI边界
        double roi_min_x = roi_center_x - roi_half_length;
        double roi_max_x = roi_center_x + roi_half_length;
        double roi_min_y = roi_center_y - roi_half_width;
        double roi_max_y = roi_center_y + roi_half_width;
        
        // 使用CropBox滤波器提取ROI区域
        pcl::CropBox<pcl::PointXYZ> crop_filter;
        crop_filter.setMin(Eigen::Vector4f(roi_min_x, roi_min_y, -100.0f, 1.0f));  // Z方向设置很大范围
        crop_filter.setMax(Eigen::Vector4f(roi_max_x, roi_max_y, 100.0f, 1.0f));
        crop_filter.setInputCloud(input_cloud);
        crop_filter.filter(*output_cloud);
        
        // 记录ROI参数用于可视化
        roi_min_x_ = roi_min_x;
        roi_max_x_ = roi_max_x;
        roi_min_y_ = roi_min_y;
        roi_max_y_ = roi_max_y;
        roi_center_z_ = (min_pt.z + max_pt.z) / 2.0;
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "ROI parameters: X[%.3f, %.3f], Y[%.3f, %.3f], Center(%.3f, %.3f)", 
                    roi_min_x, roi_max_x, roi_min_y, roi_max_y, roi_center_x, roi_center_y);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "ROI filter error: %s", e.what());
        *output_cloud = *input_cloud;
    }
}

void FeatureExtractionNode::publishROIMarker()
{
    // 创建ROI区域可视化标记
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "ascamera_hp60c_camera_link_0";
    marker.header.stamp = this->now();
    marker.ns = "roi_region";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // 设置位置（中心点）
    double center_x = (roi_min_x_ + roi_max_x_) / 2.0;
    double center_y = (roi_min_y_ + roi_max_y_) / 2.0;
    marker.pose.position.x = center_x;
    marker.pose.position.y = center_y;
    marker.pose.position.z = roi_center_z_;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    
    // 设置尺寸
    double length_x = roi_max_x_ - roi_min_x_;
    double length_y = roi_max_y_ - roi_min_y_;
    marker.scale.x = length_x;  // X方向尺寸
    marker.scale.y = length_y;  // Y方向尺寸
    marker.scale.z = 0.05;  // 增加Z方向厚度，使其更容易看到
    
    // 设置颜色（半透明绿色）
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 0.5;  // 半透明
    
    marker.lifetime = rclcpp::Duration::from_seconds(1.0);  // 增加生存时间到1秒
    
    // 发布标记
    roi_marker_pub_->publish(marker);

    // 打印调试信息
    RCLCPP_DEBUG(this->get_logger(), 
                "Published ROI marker: center (%.3f, %.3f, %.3f), size (%.3f x %.3f x %.3f)", 
                center_x, center_y, roi_center_z_, length_x, length_y, 0.05);
}

// 将点云投影到XOY平面
void FeatureExtractionNode::proj_xoy(const pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud,pcl::PointCloud<pcl::PointXYZI>::Ptr projection)
{
    // 投影到XOY平面（将Z坐标设为0）
    projection->clear();
    for (const auto& point : input_cloud->points) {
        pcl::PointXYZI proj_point;
        proj_point.x = point.x;
        proj_point.y = point.y;
        proj_point.z = 0.0;  // 投影到XOY平面，Z=0
        proj_point.intensity = point.intensity;
        projection->points.push_back(proj_point);
    }
    projection->width = projection->points.size();
    projection->height = 1;
    projection->is_dense = true;
}

// // 使用形态学方法稳定水泥特征区域
// std::pair<double, double> FeatureExtractionNode::calculateBoundingBoxWithMorphology(const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud)
// {
//     if (feature_cloud->empty()) {
//         return std::make_pair(0.0, 0.0);
//     }
    
//     try {

//         // 首先将特征点云投影到XOY平面
//         pcl::PointCloud<pcl::PointXYZI>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZI>);
//         proj_xoy(feature_cloud, projected_cloud);

//         //发布xoy点云
//         sensor_msgs::msg::PointCloud2 cloud_xoy_msg;
//         pcl::toROSMsg(*projected_cloud, cloud_xoy_msg);
//         cloud_xoy_msg.header.frame_id = "ascamera_hp60c_camera_link_0";
//         cloud_xoy_msg.header.stamp = this->now();
//         cloud_xoy_pub_->publish(cloud_xoy_msg);

//         // 1. 计算点云的边界
//         pcl::PointXYZI min_pt, max_pt;
//         pcl::getMinMax3D(*projected_cloud, min_pt, max_pt);
        
//         float x_range = max_pt.x - min_pt.x;
//         float y_range = max_pt.y - min_pt.y;
        
//         // 2. 设置图像参数 - 使用固定尺寸，而不是根据分辨率计算，总共250x50像素
//         int width = 250;  // 图像宽度，与参考代码一致
//         int height = 50;  // 图像高度，可以根据需要调整
        
//         // 3. 创建二值图像
//         cv::Mat binary_image = cv::Mat::zeros(height, width, CV_8UC1);

//         // 4. 将点云投影到图像上（使用归一化方法，与参考代码一致）
//         for (const auto& point : feature_cloud->points) {
//             // 将点云的x坐标映射到图像宽度（列）
//             int img_x = static_cast<int>((point.x - min_pt.x) / x_range * (width - 1));
            
//             // 将点云的y坐标映射到图像高度（行）
//             // 注意：点云的y对应图像的x轴，点云的z对应图像的y轴
//             // 但我们的特征点云主要是XY平面，所以这里使用x坐标映射到图像高度
//             int img_y = static_cast<int>((point.y - min_pt.y) / y_range * (height - 1));
            
//             // 确保坐标在图像范围内
//             if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
//                 binary_image.at<uchar>(img_y, img_x) = 255;
//             }
//         }
          
//         // 5. 形态学操作：先膨胀后腐蚀（闭操作），使特征更连续
//         // 使用固定核大小，而不是根据分辨率计算
//         int kernel_size = 3;  // 固定核大小
//         if (kernel_size % 2 == 0) kernel_size++;  // 确保核大小为奇数
        
//         cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
//                                                    cv::Size(kernel_size, kernel_size));
        
//         cv::Mat processed_image;
        
//         if (use_dilation_first_) {
//             // 先膨胀（连接离散点）后腐蚀（恢复大致形状）
//             cv::dilate(binary_image, processed_image, kernel, cv::Point(-1,-1), dilation_iterations_);
//             cv::erode(processed_image, processed_image, kernel, cv::Point(-1,-1), erosion_iterations_);
//         } else {
//             // 先腐蚀（去除噪声）后膨胀（恢复大小）
//             cv::erode(binary_image, processed_image, kernel, cv::Point(-1,-1), erosion_iterations_);
//             cv::dilate(processed_image, processed_image, kernel, cv::Point(-1,-1), dilation_iterations_);
//         }
        
//         // 6. 可选：显示图像（用于调试）
//         if (debug_display_image_) {
//             // // 使用cv_bridge发布图像，与参考代码一致
//             // std_msgs::msg::Header header;
//             // header.frame_id = "ascamera_hp60c_camera_link_0";
//             // header.stamp = this->now();
            
//             // // 发布二值图像
//             // sensor_msgs::msg::Image::SharedPtr binary_msg = 
//             //     cv_bridge::CvImage(header, "mono8", binary_image).toImageMsg();
//             // binary_image_pub_->publish(*binary_msg);
//             publishBinaryImage(processed_image, 
//                                "ascamera_hp60c_camera_link_0", 
//                                this->now(), 
//                                "binary_image");
//         }
        
//         // 8. 从处理后的图像中计算边界框尺寸
//         // 注意：现在需要将图像坐标转换回点云坐标
//         // 分辨率 = 实际范围 / 图像尺寸
//         float x_resolution = x_range / width;    // 每个像素在x方向的米数
//         float y_resolution = y_range / height;   // 每个像素在y方向的米数
        
//         return calculateBoundingBoxFromImage(processed_image, min_pt.x, max_pt.y, x_resolution, y_resolution);
        
//     } catch (const std::exception& e) {
//         RCLCPP_ERROR(this->get_logger(), "Error in calculateBoundingBoxWithMorphology: %s", e.what());
//         return std::make_pair(0.0, 0.0);
//     }
// }


std::pair<double, double> FeatureExtractionNode::calculateBoundingBoxWithMorphology(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud)
{
    if (feature_cloud->empty()) {
        return std::make_pair(0.0, 0.0);
    }
    
    try {
        // 首先将特征点云投影到XOY平面
        pcl::PointCloud<pcl::PointXYZI>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        proj_xoy(feature_cloud, projected_cloud);

        // 发布xoy点云
        sensor_msgs::msg::PointCloud2 cloud_xoy_msg;
        pcl::toROSMsg(*projected_cloud, cloud_xoy_msg);
        cloud_xoy_msg.header.frame_id = "ascamera_hp60c_camera_link_0";
        cloud_xoy_msg.header.stamp = this->now();
        cloud_xoy_pub_->publish(cloud_xoy_msg);

        // 1. 计算点云的边界
        pcl::PointXYZI min_pt, max_pt;
        //pcl::getMinMax3D(*projected_cloud, min_pt, max_pt);
        
        //此处默认全是0,因为不需要用点云坐标计算尺寸
        float x_range = max_pt.x - min_pt.x;
        float y_range = max_pt.y - min_pt.y;
        
        // 2. 确定分辨率：每个像素代表的实际距离
        // 根据水泥大小合理设置，假设水泥直径约0.2-0.5米
        float resolution = image_resolution_;  // 使用参数化的分辨率，默认0.005米/像素
        
        // 3. 确定图像尺寸：设置足够大的固定尺寸
        // 确保图像尺寸足够大，能容纳水泥加上足够的边界
        int width = 200;   // 图像宽度（像素）
        int height = 200;  // 图像高度（像素）
        
        // 4. 计算图像覆盖的实际范围（米）
        float image_width_meters = width * resolution;   // 图像宽度对应的实际距离
        float image_height_meters = height * resolution; // 图像高度对应的实际距离
        
        // 5. 计算点云中心
        float cloud_center_x = (min_pt.x + max_pt.x) / 2.0f;
        float cloud_center_y = (min_pt.y + max_pt.y) / 2.0f;
        
        // 6. 计算图像左下角在实际坐标系中的位置
        // 让点云位于图像中心
        float image_origin_x = cloud_center_x - image_width_meters / 2.0f;
        float image_origin_y = cloud_center_y - image_height_meters / 2.0f;
        
        // 7. 计算图像右上角在实际坐标系中的位置
        float image_max_x = image_origin_x + image_width_meters;
        float image_max_y = image_origin_y + image_height_meters;
        
        // 记录调试信息
        RCLCPP_DEBUG(this->get_logger(), 
                    "图像参数: %dx%d 像素, 分辨率: %.6f m/px", width, height, resolution);
        RCLCPP_DEBUG(this->get_logger(), 
                    "图像覆盖范围: X [%.3f, %.3f] m, Y [%.3f, %.3f] m", 
                    image_origin_x, image_max_x, image_origin_y, image_max_y);
        RCLCPP_DEBUG(this->get_logger(), 
                    "点云范围: X [%.3f, %.3f] m, Y [%.3f, %.3f] m", 
                    min_pt.x, max_pt.x, min_pt.y, max_pt.y);
        RCLCPP_DEBUG(this->get_logger(), 
                    "点云中心: (%.3f, %.3f) m", cloud_center_x, cloud_center_y);
        
        // 8. 创建二值图像
        cv::Mat binary_image = cv::Mat::zeros(height, width, CV_8UC1);

        // 9. 将点云投影到图像上（将点云投影到图像中心）
        int point_count = 0;
        int out_of_bounds = 0;
        
        for (const auto& point : projected_cloud->points) {
            // 计算像素坐标
            // 列（水平方向）：对应点云x
            int col = static_cast<int>((point.x - image_origin_x) / resolution);
            
            // 行（垂直方向）：对应点云y
            int row = static_cast<int>((point.y-image_origin_y) / resolution);
            
            // 确保坐标在图像范围内
            if (col >= 0 && col < width && row >= 0 && row < height) {
                binary_image.at<uchar>(row, col) = 255;
                point_count++;
            } else {
                out_of_bounds++;
            }
        }
        
        if (point_count == 0) {
            RCLCPP_WARN(this->get_logger(), "没有点被投影到图像中");
            return std::make_pair(0.0, 0.0);
        }
        
        // 计算点云在图像中的覆盖率
        float coverage_ratio = static_cast<float>(point_count) / projected_cloud->size();
        RCLCPP_DEBUG(this->get_logger(), 
                    "投影统计: %d/%zu 点在图像内 (%.1f%%), %d 点在图像外",
                    point_count, projected_cloud->size(), coverage_ratio * 100.0, out_of_bounds);
        
        // 如果覆盖率太低，说明图像尺寸可能不够大
        if (coverage_ratio < 0.9f) {
            RCLCPP_WARN(this->get_logger(), "点云在图像中的覆盖率较低 (%.1f%%)，考虑增大图像尺寸或调整分辨率",
                       coverage_ratio * 100.0);
        }
        
        // 10. 形态学操作：先膨胀后腐蚀（闭操作），使特征更连续
        // 根据分辨率计算合适的核大小（约5mm）
        int kernel_size = morphology_kernel_size_;
        if (kernel_size % 2 == 0) kernel_size++;  // 确保核大小为奇数
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                                                   cv::Size(kernel_size, kernel_size));
        
        cv::Mat processed_image;
        
        if (use_dilation_first_) {
            // 先膨胀（连接离散点）后腐蚀（恢复大致形状）
            cv::dilate(binary_image, processed_image, kernel, cv::Point(-1,-1), dilation_iterations_);
            cv::erode(processed_image, processed_image, kernel, cv::Point(-1,-1), erosion_iterations_);
        } else {
            // 先腐蚀（去除噪声）后膨胀（恢复大小）
            cv::erode(binary_image, processed_image, kernel, cv::Point(-1,-1), erosion_iterations_);
            cv::dilate(processed_image, processed_image, kernel, cv::Point(-1,-1), dilation_iterations_);
        }
        
        // 11. 可选：显示图像（用于调试）
        if (debug_display_image_) {
            // 为了可视化，可以创建一个带有标记的图像
            cv::Mat display_image;
            cv::cvtColor(processed_image, display_image, cv::COLOR_GRAY2BGR);
            
            // 在图像中心画一个十字标记，表示点云中心位置
            int center_col = width / 2;
            int center_row = height / 2;
            cv::line(display_image, cv::Point(center_col - 10, center_row), 
                    cv::Point(center_col + 10, center_row), cv::Scalar(0, 0, 255), 1);
            cv::line(display_image, cv::Point(center_col, center_row - 10), 
                    cv::Point(center_col, center_row + 10), cv::Scalar(0, 0, 255), 1);
            
            // 添加文本信息
            std::string info_text = cv::format("Res: %.1fmm/px", resolution * 1000.0);
            cv::putText(display_image, info_text, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            
            // 发布可视化图像
            std_msgs::msg::Header header;
            header.frame_id = "ascamera_hp60c_camera_link_0";
            header.stamp = this->now();
            
            sensor_msgs::msg::Image::SharedPtr viz_msg = 
                cv_bridge::CvImage(header, "bgr8", display_image).toImageMsg();
            
            if (visualization_pub_1) {
                visualization_pub_1->publish(*viz_msg);
            }
        }
        
        // 12. 从处理后的图像中计算边界框尺寸
        // 注意：传递给calculateBoundingBoxFromImage的是图像覆盖的实际范围
        return calculateBoundingBoxFromImage(processed_image, 
                                           image_origin_x, image_max_y, 
                                           resolution, resolution);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in calculateBoundingBoxWithMorphology: %s", e.what());
        return std::make_pair(0.0, 0.0);
    }
}


// 从二值图像中计算边界框尺寸（使用外接圆拟合）
std::pair<double, double> FeatureExtractionNode::calculateBoundingBoxFromImage(
    const cv::Mat& binary_image, 
    float min_x, float max_y, 
    float x_resolution, float y_resolution)
{
    if (binary_image.empty() || cv::countNonZero(binary_image) == 0) {
        return std::make_pair(0.0, 0.0);
    }
    
    try {
        int width = binary_image.cols;
        int height = binary_image.rows;
        
        // 1. 寻找轮廓
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        
        // 使用RETR_EXTERNAL只获取最外层轮廓
        cv::findContours(binary_image, contours, hierarchy, 
                        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) {
            RCLCPP_DEBUG(this->get_logger(), "未找到轮廓");
            return std::make_pair(0.0, 0.0);
        }
        
        // 2. 找到最大轮廓（按面积）
        int max_contour_idx = -1;
        double max_area = 0.0;
        
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_contour_idx = i;
            }
        }
        
        if (max_contour_idx == -1) {
            RCLCPP_DEBUG(this->get_logger(), "未找到有效轮廓");
            return std::make_pair(0.0, 0.0);
        }
        
        // 3. 对最大轮廓进行最小外接圆拟合
        std::vector<cv::Point> max_contour = contours[max_contour_idx];
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(max_contour, center, radius);
        
        // 4. 计算圆的直径（像素单位）
        double diameter_pixels = radius * 2.0;
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "外接圆拟合 - 中心: (%.1f, %.1f), 半径: %.1f px, 直径: %.1f px",
                    center.x, center.y, radius, diameter_pixels);
        
        // 5. 转换为实际尺寸
        // 由于图像在X和Y方向可能使用不同的分辨率，我们需要计算平均分辨率
        // 或者根据圆的中心位置进行插值计算
        
        // 方法1：使用平均分辨率
        double avg_resolution = (x_resolution + y_resolution) / 2.0;
        double real_diameter = diameter_pixels * avg_resolution;
        
        
        // 为了可视化，可以发布轮廓和外接圆信息
        if (debug_display_image_) {
            publishContourAndCircle(binary_image, contours[max_contour_idx], center, radius);
        }
        
        RCLCPP_DEBUG(this->get_logger(),
                    "实际尺寸 - 直径: %.3f m (使用平均分辨率: %.6f m/px)",
                    real_diameter, avg_resolution);
        
        // 返回直径作为长度和宽度（因为是圆形）
        return std::make_pair(real_diameter, real_diameter);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in calculateBoundingBoxFromImage: %s", e.what());
        return std::make_pair(0.0, 0.0);
    }

}


// 发布轮廓和外接圆的可视化图像
void FeatureExtractionNode::publishContourAndCircle(
    const cv::Mat& binary_image,
    const std::vector<cv::Point>& contour,
    const cv::Point2f& circle_center,
    float circle_radius)
{
    try {
        // 创建彩色图像用于可视化
        cv::Mat color_image;
        cv::cvtColor(binary_image, color_image, cv::COLOR_GRAY2BGR);
        
        // 绘制轮廓（绿色）
        cv::drawContours(color_image, std::vector<std::vector<cv::Point>>{contour}, 
                        0, cv::Scalar(0, 255, 0), 2);
        
        // 绘制最小外接圆（红色）
        cv::circle(color_image, circle_center, static_cast<int>(circle_radius), 
                  cv::Scalar(0, 0, 255), 2);
        
        // 绘制圆心（蓝色）
        cv::circle(color_image, circle_center, 3, cv::Scalar(255, 0, 0), -1);
        
        // 添加文本信息
        std::string info_text = cv::format("Radius: %.1f px", circle_radius);
        cv::putText(color_image, info_text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // 发布可视化图像
        std_msgs::msg::Header header;
        header.frame_id = "ascamera_hp60c_camera_link_0";
        header.stamp = this->now();
        
        sensor_msgs::msg::Image::SharedPtr viz_msg = 
            cv_bridge::CvImage(header, "bgr8", color_image).toImageMsg();
        
        if (visualization_pub_2) {
            visualization_pub_2->publish(*viz_msg);
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Published contour and circle visualization");
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in publishContourAndCircle: %s", e.what());
    }
}
// // 从二值图像中计算边界框尺寸（根据分辨率反算）
// std::pair<double, double> FeatureExtractionNode::calculateBoundingBoxFromImage(const cv::Mat& binary_image, float min_x, float max_y, float x_resolution, float y_resolution)
// {
//     if (binary_image.empty() || cv::countNonZero(binary_image) == 0) {
//         return std::make_pair(0.0, 0.0);
//     }
    
//     try {
//         int width = binary_image.cols;
//         int height = binary_image.rows;
        
//         // 1. 将y方向分成N个区间（类似于原始算法）
//         const int N = 100;
//         double max_width = 0.0;
//         cv::Point left_pt_pixel(-1, -1), right_pt_pixel(-1, -1);
        
//         // 计算每个区间的y边界（像素行号）
//         std::vector<int> slice_boundaries(N + 1);
//         for (int i = 0; i <= N; ++i) {
//             slice_boundaries[i] = i * height / N;
//         }
        
//         // 2. 遍历每个区间，找到x方向最宽的位置
//         for (int i = 0; i < N; ++i) {
//             int slice_v_min = slice_boundaries[i];
//             int slice_v_max = slice_boundaries[i + 1];
            
//             // 在这个区间内统计x范围
//             int min_u = INT_MAX;
//             int max_u = -1;
//             cv::Point current_left_pt(-1, -1), current_right_pt(-1, -1);
            
//             for (int v = slice_v_min; v < slice_v_max; v++) {
//                 int row_min_u = INT_MAX;
//                 int row_max_u = -1;
//                 cv::Point row_left_pt(-1, -1), row_right_pt(-1, -1);
                
//                 // 扫描这一行
//                 for (int u = 0; u < width; u++) {
//                     if (binary_image.at<uchar>(v, u) > 0) {
//                         if (u < row_min_u) {
//                             row_min_u = u;
//                             row_left_pt = cv::Point(u, v);
//                         }
//                         if (u > row_max_u) {
//                             row_max_u = u;
//                             row_right_pt = cv::Point(u, v);
//                         }
//                     }
//                 }
                
//                 // 更新整个区间的x范围
//                 if (row_min_u < min_u) {
//                     min_u = row_min_u;
//                     current_left_pt = row_left_pt;
//                 }
//                 if (row_max_u > max_u) {
//                     max_u = row_max_u;
//                     current_right_pt = row_right_pt;
//                 }
//             }
            
//             // 计算这个区间的宽度
//             if (min_u <= max_u) {
//                 int width_pixels = max_u - min_u + 1;
                
//                 if (width_pixels > max_width) {
//                     max_width = width_pixels;
//                     left_pt_pixel = current_left_pt;
//                     right_pt_pixel = current_right_pt;
//                 }
//             }
//         }
        
//         if (max_width < 1e-6) {
//             return std::make_pair(0.0, 0.0);
//         }
        
//         // 3. 寻找最下面的点（y最小的点，对应图像中行号最大的点）
//         cv::Point bottom_pt_pixel(-1, -1);
//         for (int v = height - 1; v >= 0; v--) {
//             for (int u = 0; u < width; u++) {
//                 if (binary_image.at<uchar>(v, u) > 0) {
//                     bottom_pt_pixel = cv::Point(u, v);
//                     break;
//                 }
//             }
//             if (bottom_pt_pixel.x != -1) break;
//         }
        
//         if (bottom_pt_pixel.x == -1) {
//             return std::make_pair(0.0, 0.0);
//         }
        
//         // 4. 计算最下面点到最宽直线的距离（像素单位）
//         // 使用点到直线的距离公式
//         double short_radius_pixels = calculatePointToLineDistancePixel(
//             bottom_pt_pixel, left_pt_pixel, right_pt_pixel);
        
//         // 5. 转换为实际尺寸
//         double x_length = max_width * x_resolution;
//         double y_diameter = short_radius_pixels * 2.0 * y_resolution;
        
//         RCLCPP_DEBUG(this->get_logger(),
//                     "Image analysis results - Max width: %.0f px at row %d, Bottom point at row %d",
//                     max_width, left_pt_pixel.y, bottom_pt_pixel.y);
        
//         RCLCPP_DEBUG(this->get_logger(),
//                     "Real dimensions - Length: %.3f m, Short diameter: %.3f m",
//                     x_length, y_diameter);
        
//         return std::make_pair(x_length, y_diameter);
        
//     } catch (const std::exception& e) {
//         RCLCPP_ERROR(this->get_logger(), "Error in calculateBoundingBoxFromImage: %s", e.what());
//         return std::make_pair(0.0, 0.0);
//     }
// }


// 发布二值图像
void FeatureExtractionNode::publishBinaryImage(const cv::Mat& image, const std::string& frame_id,const rclcpp::Time& stamp,const std::string& topic_name)
{
    try {
        // 确保图像不为空
        if (image.empty()) {
            RCLCPP_WARN(this->get_logger(), "Cannot publish empty image");
            return;
        }
        
        // 创建ROS图像消息
        sensor_msgs::msg::Image ros_image;
        ros_image.header.stamp = stamp;
        ros_image.header.frame_id = frame_id;
        ros_image.height = image.rows;
        ros_image.width = image.cols;
        ros_image.encoding = "mono8";  // 二值图像使用mono8编码
        ros_image.is_bigendian = false;
        ros_image.step = image.cols * sizeof(uint8_t);  // 单通道图像
        
        // 复制图像数据
        size_t data_size = image.rows * image.cols * sizeof(uint8_t);
        ros_image.data.resize(data_size);
        memcpy(ros_image.data.data(), image.data, data_size);
        
        // 发布到相应的话题
        binary_image_pub_->publish(ros_image);

        RCLCPP_DEBUG(this->get_logger(), "Published %s: %dx%d pixels", 
                    topic_name.c_str(), image.cols, image.rows);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error publishing image: %s", e.what());
    }
}


// 优化后的边界框尺寸计算方法（水泥特征区域不稳定）
std::pair<double, double> FeatureExtractionNode::calculateBoundingBoxDimensionsOptimized(const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud)
{
    if (feature_cloud->size() < min_area_points_) {
        return std::make_pair(0.0, 0.0);
    }
    
    try {
        // 1. 复制点云并按Y坐标排序
        std::vector<pcl::PointXYZI> sorted_points(feature_cloud->begin(), feature_cloud->end());
        std::sort(sorted_points.begin(), sorted_points.end(),
            [](const pcl::PointXYZI& a, const pcl::PointXYZI& b) {
                return a.y< b.y;  // y从小到大排序
            });
        
        // 2. 确定y轴范围
        float y_min = sorted_points.front().y;
        float y_max = sorted_points.back().y;
        float y_range = y_max - y_min;
        
        if (y_range < 1e-6) {
            return std::make_pair(0.0, 0.0);
        }
        
        // 3. 将y范围等分成N个区间
        const int N = 100;
        double max_width = 0.0;
        Eigen::Vector2f left_pt, right_pt;
        
        // 预计算每个区间的X边界
        std::vector<float> slice_boundaries(N + 1);
        for (int i = 0; i <= N; ++i) {
            slice_boundaries[i] = y_min + i * y_range / N;
        }
        
        // 4. 遍历每个区间
        for (int i = 0; i < N; ++i) {
            float slice_y_min = slice_boundaries[i];
            float slice_y_max = slice_boundaries[i + 1];
            
            // 使用二分查找快速找到区间内的点
            auto lower = std::lower_bound(sorted_points.begin(), sorted_points.end(), 
                slice_y_min, [](const pcl::PointXYZI& p, float y) { return p.y < y; });
            auto upper = std::upper_bound(sorted_points.begin(), sorted_points.end(), 
                slice_y_max, [](float y, const pcl::PointXYZI& p) { return y < p.y; });
            
            // 如果区间内有点，计算x范围
            if (lower != upper) {
                float min_x = std::numeric_limits<float>::max();
                float max_x = -std::numeric_limits<float>::max();
                Eigen::Vector2f current_left_pt, current_right_pt;
                
                for (auto it = lower; it != upper; ++it) {
                    if (it->x < min_x) {
                        min_x = it->x;
                        current_left_pt = Eigen::Vector2f(it->x, it->y);
                    }
                    if (it->x > max_x) {
                        max_x = it->x;
                        current_right_pt = Eigen::Vector2f(it->x, it->y);
                    }
                }
                
                float width = max_x - min_x;
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
        
        // 5. 寻找最下面的点（已经排序，最后一个点就是y最大的）
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


// 辅助函数1：计算点到直线的距离（返回绝对值）
float FeatureExtractionNode::calculatePointToLineDistance(const Eigen::Vector2f& point, const Eigen::Vector2f& line_start, const Eigen::Vector2f& line_end)
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


// 辅助函数2：计算像素坐标系中点到直线的距离
double FeatureExtractionNode::calculatePointToLineDistancePixel(const cv::Point& point, const cv::Point& line_start, const cv::Point& line_end)
{
    // 处理垂直线或水平线的情况
    if (line_start.x == line_end.x) {
        // 垂直线
        return std::abs(point.x - line_start.x);
    }
    if (line_start.y == line_end.y) {
        // 水平线
        return std::abs(point.y - line_start.y);
    }
    
    // 一般情况：点到直线的距离公式
    // 直线方程：Ax + By + C = 0
    double A = line_end.y - line_start.y;
    double B = line_start.x - line_end.x;
    double C = line_end.x * line_start.y - line_start.x * line_end.y;
    
    // 距离 = |Ax + By + C| / sqrt(A² + B²)
    double distance = std::abs(A * point.x + B * point.y + C) / std::sqrt(A * A + B * B);
    
    return distance;
}


// 基本边界框计算方法
std::pair<double, double> FeatureExtractionNode::calculateBoundingBoxDimensions(const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud)
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



// std::pair<double, double> FeatureExtractionNode::calculateBoundingBoxDimensionsOptimized(
//     const pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud)
// {
//     if (feature_cloud->size() < min_area_points_) {
//         return std::make_pair(0.0, 0.0);
//     }
    
//     try {
//         // 1. 复制点云并按X坐标排序
//         std::vector<pcl::PointXYZI> sorted_points(feature_cloud->begin(), feature_cloud->end());
//         std::sort(sorted_points.begin(), sorted_points.end(),
//             [](const pcl::PointXYZI& a, const pcl::PointXYZI& b) {
//                 return a.x < b.x;  // X从小到大排序
//             });
        
//         // 2. 确定X轴范围
//         float x_min = sorted_points.front().x;
//         float x_max = sorted_points.back().x;
//         float x_range = x_max - x_min;
        
//         if (x_range < 1e-6) {
//             return std::make_pair(0.0, 0.0);
//         }
        
//         // 3. 将X范围等分成N个区间
//         const int N = 50;
//         double max_width = 0.0;
//         Eigen::Vector2f left_pt, right_pt;
        
//         // 预计算每个区间的X边界
//         std::vector<float> slice_boundaries(N + 1);
//         for (int i = 0; i <= N; ++i) {
//             slice_boundaries[i] = x_min + i * x_range / N;
//         }
        
//         // 4. 遍历每个区间
//         for (int i = 0; i < N; ++i) {
//             float slice_x_min = slice_boundaries[i];
//             float slice_x_max = slice_boundaries[i + 1];
            
//             // 使用二分查找快速找到区间内的点
//             auto lower = std::lower_bound(sorted_points.begin(), sorted_points.end(), 
//                 slice_x_min, [](const pcl::PointXYZI& p, float x) { return p.x < x; });
//             auto upper = std::upper_bound(sorted_points.begin(), sorted_points.end(), 
//                 slice_x_max, [](float x, const pcl::PointXYZI& p) { return x < p.x; });
            
//             // 如果区间内有点，计算Y范围
//             if (lower != upper) {
//                 float min_y = std::numeric_limits<float>::max();
//                 float max_y = -std::numeric_limits<float>::max();
//                 Eigen::Vector2f current_left_pt, current_right_pt;
                
//                 for (auto it = lower; it != upper; ++it) {
//                     if (it->y < min_y) {
//                         min_y = it->y;
//                         current_left_pt = Eigen::Vector2f(it->x, it->y);
//                     }
//                     if (it->y > max_y) {
//                         max_y = it->y;
//                         current_right_pt = Eigen::Vector2f(it->x, it->y);
//                     }
//                 }
                
//                 float width = max_y - min_y;
//                 if (width > max_width) {
//                     max_width = width;
//                     left_pt = current_left_pt;
//                     right_pt = current_right_pt;
//                 }
//             }
//         }
        
//         if (max_width < 1e-6) {
//             return std::make_pair(0.0, 0.0);
//         }
        
//         // 5. 寻找最下面的点（已经排序，最后一个点就是X最大的）
//         Eigen::Vector2f bottom_point(sorted_points.back().x, sorted_points.back().y);
        
//         // 6. 计算短直径
//         float short_radius = calculatePointToLineDistance(bottom_point, left_pt, right_pt);
//         double short_diameter = short_radius * 2.0;
        
//         return std::make_pair(max_width, short_diameter);
        
//     } catch (const std::exception& e) {
//         RCLCPP_ERROR(this->get_logger(), "Error: %s", e.what());
//         return std::make_pair(0.0, 0.0);
//     }
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