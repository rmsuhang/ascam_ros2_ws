#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>

class StreamManager : public rclcpp::Node {
public:
    StreamManager(const std::string& node_name, const rclcpp::NodeOptions& options);
    
    void set_current_frame(const cv::Mat& frame);
    cv::Mat get_current_frame() const;
    bool has_frame() const;
    
private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    std::string image_topic_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    cv::Mat current_frame_;
    mutable std::mutex frame_mutex_;
    std::atomic<bool> has_frame_{false};
    
};