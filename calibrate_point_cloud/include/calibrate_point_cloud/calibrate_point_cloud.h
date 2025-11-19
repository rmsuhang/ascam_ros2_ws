#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include <fstream>
#include <yaml-cpp/yaml.h>

using SetParametersResult = rcl_interfaces::msg::SetParametersResult;

class FilteringNode : public rclcpp::Node
{
public:
    FilteringNode(const std::string &node_name);

public:
    enum FilterAxes
    {
        kXAxis = 0,
        kYAxis,
        kZAxis
    };

    struct AxisRange
    {
        double min;
        double max;

        bool operator==(const AxisRange &other) const
        {
            return min == other.min && max == other.max;
        }
    };
    struct FilterArea
    {
        AxisRange x;
        AxisRange y;
        AxisRange z;

        bool operator==(const FilterArea &other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

private:
    using FilterCloudPtr = pcl::PointCloud<pcl::PointXYZI>::Ptr;
    using SubPtr = rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr;
    using PubPtr = rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr;
    using ImagePubPtr = rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr;

    void rslidar1Callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void rslidar2Callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    void timerCallback();
    SetParametersResult parametersSetCallback(const std::vector<rclcpp::Parameter> &parameters);

    // 直通滤波
    void passThroughFilter(FilterCloudPtr cloud, FilterCloudPtr &cloud_medium,
                           FilterAxes axis, double min, double max);

    // 配准
    void performRegistration(FilterCloudPtr src_cloud,
                             FilterCloudPtr tgt_cloud,
                             FilterCloudPtr registration_cloud);
    void performFilter(FilterCloudPtr cloud, FilterCloudPtr cloud_voxel);

    // yoz面投影
    void proj_yoz(FilterCloudPtr cloud, FilterCloudPtr projection);

    // 保存投影图像
    void ToImage(FilterCloudPtr cloud);

    // 发布滤波、配准后的点云数据
    void publishCloud(PubPtr pub, FilterCloudPtr cloud);

    // 加载yaml文件
    void loadParamsFromYaml(const std::string &filename);

    // 辅助函数：将YAML节点转换为FilterArea
    FilterArea nodeToFilterArea(const YAML::Node &node);

    // 辅助函数：将FilterArea转换为YAML节点
    YAML::Node filterAreaToNode(const FilterArea &area);

    void updateAndSaveParams(const std::string &filename,
                             const FilterArea &new_rslidar1,
                             const FilterArea &new_rslidar2);

    void loadDefaultParams();

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
    // PCL 点云
    FilterCloudPtr rslidar1_cloud_;
    FilterCloudPtr rslidar2_cloud_;

    // 定时器
    rclcpp::TimerBase::SharedPtr timer_;

    // 订阅雷达原始数据
    SubPtr rslidar1_subscribe_;
    SubPtr rslidar2_subscribe_;

    // 发布配准数据
    PubPtr lidar1_voxel_publisher_;
    PubPtr lidar2_voxel_publisher_;
    PubPtr registration_publisher_;
    PubPtr yoz_cloud_publisher_;
    ImagePubPtr image_publisher_;

    std::atomic_bool rslidar1_ready_;
    std::atomic_bool rslidar2_ready_;

    // 参数名称常量（避免硬编码）
    const std::string PARAM_RSLIDAR1 = "rsliadr1_filter_area";
    const std::string PARAM_RSLIDAR2 = "rsliadr2_filter_area";

    // 参数
    FilterArea rsliadr1_area_;
    FilterArea rsliadr2_area_;

    std::mutex mutex_;
};
