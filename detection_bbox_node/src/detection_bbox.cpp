#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <mutex>
#include <fstream>

using sensor_msgs::msg::PointCloud2;
using std::placeholders::_1;
using vision_msgs::msg::Detection2DArray;
using FilterCloudPtr = pcl::PointCloud<pcl::PointXYZI>::Ptr;

class DetectionSubscriber : public rclcpp::Node
{
    struct BumperCoords
    {
        std::string name;
        int x1, y1, x2, y2;
        float score;
    };

private:
    rclcpp::Subscription<Detection2DArray>::SharedPtr subscri_ption_;
    rclcpp::Subscription<PointCloud2>::SharedPtr subscri_yozcloud_;
    std::map<std::string, BumperCoords> bumper_map;

    FilterCloudPtr yoz_cloud_;
    std::mutex mutex_;

public:
    DetectionSubscriber() : Node("detection_bbox")
    {
        // 创建订阅器，订阅 /yolo_result 话题
        subscri_ption_ = this->create_subscription<Detection2DArray>(
            "yolo_result", 10,
            std::bind(&DetectionSubscriber::detection_callback, this, _1));

        subscri_yozcloud_ = this->create_subscription<PointCloud2>(
            "/cloud_yoz", rclcpp::SensorDataQoS{},
            [this](const PointCloud2::SharedPtr msg)
            {
                FilterCloudPtr cloud{new pcl::PointCloud<pcl::PointXYZI>{}};
                pcl::fromROSMsg(*msg, *cloud);
                std::lock_guard<std::mutex> lock(mutex_); // 加锁
                yoz_cloud_ = cloud;
            });

        //RCLCPP_INFO(this->get_logger(), "C++ 订阅器已启动，等待检测结果...");
    }

private:

    // 写入函数
    void updateYamlFile(const std::string &key, double value)
    {
        std::string package_dir = ament_index_cpp::get_package_share_directory("process_point_cloud");
        std::string file_path = package_dir + "/config/process_node_config.yaml";
        YAML::Node config = YAML::LoadFile(file_path);
        
        char buffer[32];
        if (config[key]) {
            snprintf(buffer, sizeof(buffer), "%.2f", std::round(value * 100) / 100);
            config[key] = buffer;
        }

        // 写回文件
        std::ofstream fout(file_path);
        fout << config;
        fout.close();

        YAML::Node updated_config = YAML::LoadFile(file_path);
        if (updated_config[key].as<std::string>() == std::string(buffer)) {
            RCLCPP_INFO(this->get_logger(), "校准完成！校准位置保存成功");
        }
    }

    // 回调函数：处理接收到的检测结果
    void detection_callback(const Detection2DArray::SharedPtr msg)
    {
        //RCLCPP_INFO(this->get_logger(), "接受到识别结果，触发回调");

        bumper_map.clear();
        for (const auto &detection : msg->detections)
        {
            // 提取边框坐标
            float center_x, center_y;
            float size_x = detection.bbox.size_x;
            float size_y = detection.bbox.size_y;

            center_x = detection.bbox.center.position.x;
            center_y = detection.bbox.center.position.y;

            // 计算左上角右下角下标
            int x1 = static_cast<int>(center_x - size_x / 2);
            int y1 = static_cast<int>(center_y - size_y / 2);
            int x2 = static_cast<int>(center_x + size_x / 2);
            int y2 = static_cast<int>(center_y + size_y / 2);

            // 提取类别和置信度
            for (const auto &hypothesis : detection.results)
            {
                std::string class_id = hypothesis.hypothesis.class_id;
                float score = hypothesis.hypothesis.score;

                // 只处理高置信度的结果
                if (score > 0.5f && (class_id == "front bumper" || class_id == "rear bumper"))
                {
                    BumperCoords bumperCoords{class_id, x1, y1, x2, y2, score};
                    bumper_map.insert({class_id, bumperCoords});
                }
            }
        }

        pcl::PointXYZI min_pt, max_pt;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pcl::getMinMax3D(*yoz_cloud_, min_pt, max_pt);
        }
        int width = 1000; // 图像宽度
        // int height = 200; // 图像高度
        pcl::PointXYZI front_bbox_min, front_bbox_max;
        pcl::PointXYZI reor_bbox_min, reor_bbox_max;

        // 1. 反转垂直翻转
        // int unflipped_y1 = height - 1 - bumperCoords.y2;
        // int unflipped_y2 = height - 1 - bumperCoords.y1;

        // 2. 转换回点云坐标
        float range_y = max_pt.y - min_pt.y;
        // float range_z = max_pt.z - min_pt.z;

        auto front_it = bumper_map.find("front bumper"); // 返回前挡板节点迭代器
        if (front_it != bumper_map.end())
        {
            front_bbox_min.y = (static_cast<float>(front_it->second.x1) / (width - 1)) * range_y + min_pt.y;//min_pt.y是基准，min_pt.y所在坐标系是上方雷达坐标系
            front_bbox_max.y = (static_cast<float>(front_it->second.x2) / (width - 1)) * range_y + min_pt.y;

            this->updateYamlFile("stop_position", front_bbox_min.y);
            RCLCPP_INFO(this->get_logger(), "车厢前挡板位置：%f", front_bbox_min.y);
        }

        auto reor_it = bumper_map.find("rear bumper"); // 返回后挡板节点迭代器
        if (reor_it != bumper_map.end())
        {
            reor_bbox_min.y = (static_cast<float>(reor_it->second.x1) / (width - 1)) * range_y + min_pt.y;
            reor_bbox_max.y = (static_cast<float>(reor_it->second.x2) / (width - 1)) * range_y + min_pt.y;
            float rear_length = reor_bbox_min.y - min_pt.y;
            //this->updateYamlFile("rear_length", rear_length);
            RCLCPP_INFO(this->get_logger(), "车厢后挡板位置：%f", rear_length);//这个是后挡板到后障碍物的距离，表明车走了多远
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DetectionSubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}