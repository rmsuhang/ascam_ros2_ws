#include "calibrate_point_cloud/calibrate_point_cloud.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <pcl/filters/project_inliers.h>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/opencv.hpp>
#include <pcl/io/ply_io.h>
#include <filesystem>
#include <chrono>
#include <string>
#include <sys/stat.h>
#include <ctime>
#include <unistd.h>
#include <mutex>
#include "rclcpp/logging.hpp"

FilteringNode::FilteringNode(const std::string &node_name)
    : Node{node_name}, rslidar1_cloud_{new pcl::PointCloud<pcl::PointXYZI>{}}, rslidar2_cloud_{new pcl::PointCloud<pcl::PointXYZI>{}}, rslidar1_ready_{false}, rslidar2_ready_{false}
{

    rslidar1_subscribe_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/rslidar_points1", rclcpp::QoS(rclcpp::KeepLast(5)),
        std::bind(&FilteringNode::rslidar1Callback, this, std::placeholders::_1));
    // RCLCPP_INFO(get_logger(), "create rslidar1.");

    rslidar2_subscribe_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/rslidar_points2", rclcpp::QoS(rclcpp::KeepLast(5)),
        std::bind(&FilteringNode::rslidar2Callback, this, std::placeholders::_1));
    // RCLCPP_INFO(get_logger(), "create rslidar2.");

    using namespace std::chrono_literals;
    timer_ = create_wall_timer(100ms, std::bind(&FilteringNode::timerCallback, this));

    lidar1_voxel_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/cloud1_voxel", rclcpp::QoS(rclcpp::KeepLast(5)));
    lidar2_voxel_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/cloud2_voxel", rclcpp::QoS(rclcpp::KeepLast(5)));
    registration_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/cloud_registration", rclcpp::QoS(rclcpp::KeepLast(5)));
    yoz_cloud_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/cloud_yoz", rclcpp::QoS(rclcpp::KeepLast(5)));

    image_publisher_ = create_publisher<sensor_msgs::msg::Image>(
        "/cloud_image", rclcpp::QoS(rclcpp::KeepLast(5)));

    // 声明参数（带默认值）
    this->declare_parameter(PARAM_RSLIDAR1 + ".x_axis.min", -2.3);
    this->declare_parameter(PARAM_RSLIDAR1 + ".x_axis.max", 5.0);

    this->declare_parameter(PARAM_RSLIDAR1 + ".y_axis.min", -13.8);
    this->declare_parameter(PARAM_RSLIDAR1 + ".y_axis.max", 19.2);

    this->declare_parameter(PARAM_RSLIDAR1 + ".z_axis.min", 0.0);
    this->declare_parameter(PARAM_RSLIDAR1 + ".z_axis.max", 4.5);

    this->declare_parameter(PARAM_RSLIDAR2 + ".x_axis.min", -2.3);
    this->declare_parameter(PARAM_RSLIDAR2 + ".x_axis.max", 5.0);

    this->declare_parameter(PARAM_RSLIDAR2 + ".y_axis.min", -13.8);
    this->declare_parameter(PARAM_RSLIDAR2 + ".y_axis.max", 19.2);

    this->declare_parameter(PARAM_RSLIDAR2 + ".z_axis.min", 0.0);
    this->declare_parameter(PARAM_RSLIDAR2 + ".z_axis.max", 4.5);

    // 获取yaml配置参数
    std::string package_dir = ament_index_cpp::get_package_share_directory("process_point_cloud");
    std::string yaml_path = package_dir + "/config/process_node_config.yaml";
    RCLCPP_INFO(this->get_logger(), "load YAML FIle path: %s", yaml_path.c_str());

    loadParamsFromYaml(yaml_path);
    RCLCPP_INFO(this->get_logger(), "rsliadr1_area_x_min: %f", rsliadr1_area_.x.min);
    RCLCPP_INFO(this->get_logger(), "rsliadr1_area_x_max: %f", rsliadr1_area_.x.max);
    RCLCPP_INFO(this->get_logger(), "rsliadr1_area_y_min: %f", rsliadr1_area_.y.min);
    RCLCPP_INFO(this->get_logger(), "rsliadr1_area_y_max: %f", rsliadr1_area_.y.max);
    RCLCPP_INFO(this->get_logger(), "rsliadr1_area_z_min: %f", rsliadr1_area_.z.min);
    RCLCPP_INFO(this->get_logger(), "rsliadr1_area_z_max: %f", rsliadr1_area_.z.max);

    RCLCPP_INFO(this->get_logger(), "rsliadr2_area_x_min: %f", rsliadr2_area_.x.min);
    RCLCPP_INFO(this->get_logger(), "rsliadr2_area_x_max: %f", rsliadr2_area_.x.max);
    RCLCPP_INFO(this->get_logger(), "rsliadr2_area_y_min: %f", rsliadr2_area_.y.min);
    RCLCPP_INFO(this->get_logger(), "rsliadr2_area_y_max: %f", rsliadr2_area_.y.max);
    RCLCPP_INFO(this->get_logger(), "rsliadr2_area_z_min: %f", rsliadr2_area_.z.min);
    RCLCPP_INFO(this->get_logger(), "rsliadr2_area_z_max: %f", rsliadr2_area_.z.max);

    // 绑定参数回调
    params_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&FilteringNode::parametersSetCallback, this, std::placeholders::_1));
}

void FilteringNode::rslidar1Callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    pcl::fromROSMsg(*msg, *rslidar1_cloud_);
    rslidar1_ready_.store(true);
}

void FilteringNode::rslidar2Callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    pcl::fromROSMsg(*msg, *rslidar2_cloud_);
    rslidar2_ready_.store(true);
}

void FilteringNode::timerCallback()
{
    if (!rslidar1_ready_ || !rslidar2_ready_)
    {
        return;
    }

    FilterCloudPtr cloud1_voxel{new pcl::PointCloud<pcl::PointXYZI>{}};
    FilterCloudPtr cloud2_voxel{new pcl::PointCloud<pcl::PointXYZI>{}};

    passThroughFilter(rslidar1_cloud_, cloud1_voxel, kXAxis, rsliadr1_area_.x.min, rsliadr1_area_.x.max);
    passThroughFilter(cloud1_voxel, cloud1_voxel, kYAxis, rsliadr1_area_.y.min, rsliadr1_area_.y.max);
    passThroughFilter(cloud1_voxel, cloud1_voxel, kZAxis, rsliadr1_area_.z.min, rsliadr1_area_.z.max);
    publishCloud(lidar1_voxel_publisher_, cloud1_voxel);

    passThroughFilter(rslidar2_cloud_, cloud2_voxel, kXAxis, rsliadr2_area_.x.min, rsliadr2_area_.x.max);
    passThroughFilter(cloud2_voxel, cloud2_voxel, kYAxis, rsliadr2_area_.y.min, rsliadr2_area_.y.max);
    passThroughFilter(cloud2_voxel, cloud2_voxel, kZAxis, rsliadr2_area_.z.min, rsliadr2_area_.z.max);
    publishCloud(lidar2_voxel_publisher_, cloud2_voxel);

    // 发布配准点云
    FilterCloudPtr registration_cloud{new pcl::PointCloud<pcl::PointXYZI>{}};
    performRegistration(cloud1_voxel, cloud2_voxel, registration_cloud);
    performFilter(registration_cloud, registration_cloud);
    publishCloud(registration_publisher_, registration_cloud);

    // 发布yoz投影点云
    FilterCloudPtr cloud_yoz{new pcl::PointCloud<pcl::PointXYZI>{}};
    proj_yoz(registration_cloud, cloud_yoz);
    publishCloud(yoz_cloud_publisher_, cloud_yoz);

    // 保存yoz投影图像
    ToImage(cloud_yoz);

    std::lock_guard<std::mutex> lock(mutex_);
    rslidar1_ready_.store(false);
    rslidar2_ready_.store(false);
}

SetParametersResult FilteringNode::parametersSetCallback(const std::vector<rclcpp::Parameter> &parameters)
{

    auto result = rcl_interfaces::msg::SetParametersResult();
    result.successful = true;

    for (const auto &param : parameters)
    {

        if (param.get_name().find(PARAM_RSLIDAR1) != std::string::npos)
        {
            // 解析参数名并更新对应值
            if (param.get_name().find(".x_axis.min") != std::string::npos)
            {
                rsliadr1_area_.x.min = param.as_double();
            }
            else if (param.get_name().find(".x_axis.max") != std::string::npos)
            {
                rsliadr1_area_.x.max = param.as_double();
            }
            else if (param.get_name().find(".y_axis.min") != std::string::npos)
            {
                rsliadr1_area_.y.min = param.as_double();
            }
            else if (param.get_name().find(".y_axis.max") != std::string::npos)
            {
                rsliadr1_area_.y.max = param.as_double();
            }
            else if (param.get_name().find(".z_axis.min") != std::string::npos)
            {
                rsliadr1_area_.z.min = param.as_double();
            }
            else if (param.get_name().find(".z_axis.max") != std::string::npos)
            {
                rsliadr1_area_.z.max = param.as_double();
            }
        }
        else if (param.get_name().find(PARAM_RSLIDAR2) != std::string::npos)
        {
           if (param.get_name().find(".x_axis.min") != std::string::npos)
            {
                rsliadr2_area_.x.min = param.as_double();
            }
            else if (param.get_name().find(".x_axis.max") != std::string::npos)
            {
                rsliadr2_area_.x.max = param.as_double();
            }
            else if (param.get_name().find(".y_axis.min") != std::string::npos)
            {
                rsliadr2_area_.y.min = param.as_double();
            }
            else if (param.get_name().find(".y_axis.max") != std::string::npos)
            {
                rsliadr2_area_.y.max = param.as_double();
            }
            else if (param.get_name().find(".z_axis.min") != std::string::npos)
            {
                rsliadr2_area_.z.min = param.as_double();
            }
            else if (param.get_name().find(".z_axis.max") != std::string::npos)
            {
                rsliadr2_area_.z.max = param.as_double();
            }
        }
    }

    // 点云工程功能包下的yaml文件
    std::string package_dir = ament_index_cpp::get_package_share_directory("process_point_cloud");
    std::string yaml_path = package_dir + "/config/process_node_config.yaml";
    // RCLCPP_INFO(this->get_logger(), "updete YAML FIle path: %s", yaml_path.c_str());
    updateAndSaveParams(yaml_path, rsliadr1_area_, rsliadr2_area_);
    return result;
}

void FilteringNode::passThroughFilter(FilterCloudPtr cloud, FilterCloudPtr &cloud_medium,
                                      FilterAxes axis, double min, double max)
{
    pcl::PassThrough<pcl::PointXYZI> pass{};
    pass.setInputCloud(cloud); // 待滤波点云

    // 滤波轴方向
    switch (axis)
    {
    case kXAxis:
        pass.setFilterFieldName("x");
        break;
    case kYAxis:
        pass.setFilterFieldName("y");
        break;
    case kZAxis:
        pass.setFilterFieldName("z");
        break;
    default:
        break;
    }

    pass.setNegative(false); // false: 保留当前范围内数据, true: 取反
    pass.setFilterLimits(min, max);
    pass.filter(*cloud_medium); // 转存滤波结果
}

void FilteringNode::performRegistration(FilterCloudPtr src_cloud,
                                        FilterCloudPtr tgt_cloud,
                                        FilterCloudPtr registration_cloud)
{
    if (!src_cloud || !tgt_cloud || src_cloud->empty() || tgt_cloud->empty())
    {
        RCLCPP_ERROR(get_logger(), "点云数据为空！");
        return;
    }

    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    icp.setInputSource(src_cloud);
    icp.setInputTarget(tgt_cloud);

    // 设置参数
    icp.setMaxCorrespondenceDistance(0.1);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(0.25);

    icp.align(*registration_cloud);
    *registration_cloud = *registration_cloud + *tgt_cloud;

    if (!icp.hasConverged())
    {
        RCLCPP_ERROR(get_logger(), "ICP 配准失败");
    }
}

void FilteringNode::performFilter(FilterCloudPtr cloud, FilterCloudPtr cloud_voxel)
{
    // 矩形的最大和最小边界
    Eigen::Vector4f min_point{};
    min_point << 0.2 - 1.7 / 2, 1 - 1.5 / 2, 3.5 - 2 / 2, 1.0; // 最小边界点
    Eigen::Vector4f max_point{};
    max_point << 0.2 + 1.7 / 2, 1 + 1.5 / 2, 3.5 + 2 / 2, 1.0; // 最大边界点

    // 过滤器
    pcl::CropBox<pcl::PointXYZI> box_filter{};
    box_filter.setMin(min_point);
    box_filter.setMax(max_point);
    box_filter.setNegative(true); // 反向滤波，保留矩形框外侧的点
    box_filter.setInputCloud(cloud);
    box_filter.filter(*cloud_voxel); // 转存过滤后的点云
}

void FilteringNode::proj_yoz(FilterCloudPtr cloud, FilterCloudPtr projection)
{
    // 投影到YOZ平面
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients); // 创建投影平面系数
    coefficients->values.resize(4);                                       // x轴法向量投影在ros2中如何将沿x轴方向的长度为5的点云数据去除
    coefficients->values[0] = 1.0;
    coefficients->values[1] = 0.0;
    coefficients->values[2] = 0.0;
    coefficients->values[3] = 0.0; // 原点距离

    pcl::ProjectInliers<pcl::PointXYZI> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud);
    proj.setModelCoefficients(coefficients);
    proj.filter(*projection);
}

void FilteringNode::ToImage(FilterCloudPtr cloud)
{
    // 创建图像，假设点云的范围在[0, 1]之间
    int width = 1000; // 图像宽度
    int height = 200; // 图像高度
    cv::Mat image(height, width, CV_8UC1, cv::Scalar(0));

    pcl::PointXYZI min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    float range_y = max_pt.y - min_pt.y;
    float range_z = max_pt.z - min_pt.z;
    // RCLCPP_INFO(this->get_logger(),"Max_pt.y: %f, Min_pt.y: %f",max_pt.y,min_pt.y);
    // RCLCPP_INFO(this->get_logger(),"Max_pt.z: %f, Min_pt.z: %f",max_pt.z, min_pt.z);
    // RCLCPP_INFO(this->get_logger(),"range_y: %f, range_z: %f",range_y, range_z);
    // 将点云转换为图像
    for (const auto &point : cloud->points)
    {
        int x = static_cast<int>((point.y - min_pt.y) / range_y * (width - 1));
        int y = static_cast<int>((point.z - min_pt.z) / range_z * (height - 1));

        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            image.at<uchar>(y, x) = 255; // 将点映射到图像上，设置为白色
        }
    }

    // 垂直180度镜像
    cv::Mat rotated_image;
    cv::flip(image, rotated_image, 0);

    // // 保存PNG图像（带时间戳）
    // time_t now = time(nullptr);
    // struct tm *timeinfo = localtime(&now);
    // char time_str[20];
    // strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", timeinfo);

    // std::string save_dir;
    // std::string pkg_path = ament_index_cpp::get_package_share_directory("calibrate_point_cloud");
    // save_dir = pkg_path + "/output/";

    // struct stat st;
    // if (stat(save_dir.c_str(), &st) == -1)
    // {
    //     mkdir(save_dir.c_str(), 0755); // 创建目录
    // }

    // std::string filename = save_dir + time_str + ".png";
    // cv::imwrite(filename, rotated_image);

    // 使用 cv_bridge 将 OpenCV 图像转换为 ROS2 图像消息
    std_msgs::msg::Header header;
    header.frame_id = "rslidar"; // 设置适当的坐标系
    header.stamp = this->now();
    sensor_msgs::msg::Image::SharedPtr image_msg = cv_bridge::CvImage(header, "mono8", rotated_image).toImageMsg();

    // 发布图像消息
    image_publisher_->publish(*image_msg);
}

void FilteringNode::publishCloud(PubPtr pub, FilterCloudPtr cloud)
{
    sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg{new sensor_msgs::msg::PointCloud2{}};
    pcl::toROSMsg(*cloud, *cloud_msg);
    cloud_msg->header.frame_id = "rslidar";
    cloud_msg->header.stamp = now();
    pub->publish(*cloud_msg);
}

void FilteringNode::loadParamsFromYaml(const std::string &filename)
{
    try
    {
        YAML::Node config = YAML::LoadFile(filename);
        if (config[PARAM_RSLIDAR1])
        {
            rsliadr1_area_ = nodeToFilterArea(config[PARAM_RSLIDAR1]);
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR1 + ".x_axis.min", rsliadr1_area_.x.min));
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR1 + ".x_axis.max", rsliadr1_area_.x.max));
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR1 + ".y_axis.min", rsliadr1_area_.y.min));
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR1 + ".y_axis.max", rsliadr1_area_.y.max));
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR1 + ".z_axis.min", rsliadr1_area_.z.min));
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR1 + ".z_axis.max", rsliadr1_area_.z.max));
        }
        if (config[PARAM_RSLIDAR2])
        {
            rsliadr2_area_ = nodeToFilterArea(config[PARAM_RSLIDAR2]);

            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR2 + ".x_axis.min", rsliadr2_area_.x.min));
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR2 + ".x_axis.max", rsliadr2_area_.x.max));
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR2 + ".y_axis.min", rsliadr2_area_.y.min));
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR2 + ".y_axis.max", rsliadr2_area_.y.max));
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR2 + ".z_axis.min", rsliadr2_area_.z.min));
            this->set_parameter(rclcpp::Parameter(
                PARAM_RSLIDAR2 + ".z_axis.max", rsliadr2_area_.z.max));
        }
        RCLCPP_INFO(this->get_logger(), "配置文件读取成功！");
    }
    catch (const YAML::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "目标位置配置文件为空");
        loadDefaultParams();
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "加载配置文件失败: %s", e.what());
        loadDefaultParams();
    }
}

YAML::Node FilteringNode::filterAreaToNode(const FilterArea &area)
{
    YAML::Node node;
    node["x_axis"]["min"] = std::round(area.x.min * 100) / 100;
    node["x_axis"]["max"] = std::round(area.x.max * 100) / 100;

    node["y_axis"]["min"] = std::round(area.y.min * 100) / 100;
    node["y_axis"]["max"] = std::round(area.y.max * 100) / 100;

    node["z_axis"]["min"] = std::round(area.z.min * 100) / 100;
    node["z_axis"]["max"] = std::round(area.z.max * 100) / 100;
    return node;
}

FilteringNode::FilterArea FilteringNode::nodeToFilterArea(const YAML::Node &node)
{
    FilterArea area;

    if (node["x_axis"] && node["x_axis"].size() == 2)
    {
        area.x.min = node["x_axis"]["min"].as<double>();
        area.x.max = node["x_axis"]["max"].as<double>();
    }

    if (node["y_axis"] && node["y_axis"].size() == 2)
    {
        area.y.min = node["y_axis"]["min"].as<double>();
        area.y.max = node["y_axis"]["max"].as<double>();
    }

    if (node["z_axis"] && node["z_axis"].size() == 2)
    {
        area.z.min = node["z_axis"]["min"].as<double>();
        area.z.max = node["z_axis"]["max"].as<double>();
    }

    return area;
}

void FilteringNode::updateAndSaveParams(const std::string &filename,
                                        const FilterArea &new_rslidar1, const FilterArea &new_rslidar2)
{
    std::unordered_map<std::string, YAML::Node> all_params;
    YAML::Node yaml_file = YAML::LoadFile(filename);
    for (YAML::const_iterator it = yaml_file.begin(); it != yaml_file.end(); ++it)
    {
        std::string param_name = it->first.as<std::string>();
        all_params[param_name] = it->second;
    }

    all_params[PARAM_RSLIDAR1] = filterAreaToNode(new_rslidar1);
    all_params[PARAM_RSLIDAR2] = filterAreaToNode(new_rslidar2);

    YAML::Emitter emitter;
    emitter.SetIndent(2);              // 标准缩进
    emitter.SetMapFormat(YAML::Block); // 块格式
    emitter.SetSeqFormat(YAML::Block); // 序列换行
    emitter.SetBoolFormat(YAML::TrueFalseBool);
    emitter << YAML::BeginMap;

    std::vector<std::string> sorted_keys;
    for (const auto &pair : all_params)
    {
        sorted_keys.push_back(pair.first);
    }
    std::sort(sorted_keys.begin(), sorted_keys.end());

    for (const auto &key : sorted_keys)
    {
        emitter << YAML::Key << key << YAML::Value << all_params[key];
    }

    emitter << YAML::EndMap;

    std::ofstream fout(filename);
    fout << emitter.c_str();
    fout.close();

    YAML::Node config = YAML::LoadFile(filename);
    FilterArea area_rslidar1 = nodeToFilterArea(config[PARAM_RSLIDAR1]);
    FilterArea area_rslidar2 = nodeToFilterArea(config[PARAM_RSLIDAR2]);
    if (area_rslidar1 == new_rslidar1 && area_rslidar2 == new_rslidar2)
    {
        RCLCPP_INFO(this->get_logger(), "修改成功！");
    }
}

void FilteringNode::loadDefaultParams()
{
    auto loadAxisRange = [this](const std::string &prefix, FilterArea &area)
    {
        area.x.min = this->get_parameter(prefix + ".x_axis.min").as_double();
        area.x.max = this->get_parameter(prefix + ".x_axis.max").as_double();

        area.y.min = this->get_parameter(prefix + ".y_axis.min").as_double();
        area.y.max = this->get_parameter(prefix + ".y_axis.max").as_double();

        area.z.min = this->get_parameter(prefix + ".z_axis.min").as_double();
        area.z.max = this->get_parameter(prefix + ".z_axis.max").as_double();
    };

    loadAxisRange(PARAM_RSLIDAR1, rsliadr1_area_);
    loadAxisRange(PARAM_RSLIDAR2, rsliadr2_area_);
}