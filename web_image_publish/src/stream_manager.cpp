#include "web_image_publish/stream_manager.hpp"
#include <image_transport/image_transport.hpp>

StreamManager::StreamManager(const std::string& node_name, 
                           const rclcpp::NodeOptions& options)
    : Node(node_name, options) {
    
    this->declare_parameter("image_topic", "/ascamera_hp60c/camera_publisher/rgb0/image");
    image_topic_ = this->get_parameter("image_topic").as_string();
    
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic_, 
        rclcpp::SensorDataQoS().reliable(),
        [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
            this->image_callback(msg);
        });

    
    RCLCPP_INFO(this->get_logger(), "Stream manager subscribed to: %s", 
                image_topic_.c_str());
}

void StreamManager::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        
        std::lock_guard<std::mutex> lock(frame_mutex_);
        current_frame_ = cv_ptr->image.clone();
        has_frame_ = true;
        
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void StreamManager::set_current_frame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    current_frame_ = frame.clone();
    has_frame_ = true;
}

cv::Mat StreamManager::get_current_frame() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return current_frame_.clone();
}

bool StreamManager::has_frame() const {
    return has_frame_;
}