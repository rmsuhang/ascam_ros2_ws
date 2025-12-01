#include "web_image_publish/http_server.hpp"
#include "web_image_publish/stream_manager.hpp"
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <thread>
#include <vector>

class VideoStreamServer : public rclcpp::Node {
public:
    VideoStreamServer() : Node("video_stream_server") {
        // 声明参数
        this->declare_parameter("image_topic", "/ascamera_hp60c/camera_publisher/rgb0/image");
        this->declare_parameter("port", 8080);
        this->declare_parameter("frame_rate", 30);
        this->declare_parameter("jpeg_quality", 80);
        this->declare_parameter("thread_count", 4);
        
        // 获取参数
        image_topic_ = this->get_parameter("image_topic").as_string();
        port_ = this->get_parameter("port").as_int();
        frame_rate_ = this->get_parameter("frame_rate").as_int();
        jpeg_quality_ = this->get_parameter("jpeg_quality").as_int();
        thread_count_ = this->get_parameter("thread_count").as_int();
        
        // 创建流管理器
        rclcpp::NodeOptions options;
        options.arguments({
            "--ros-args", 
            "-r", "__node:=stream_manager", 
            "-p", "image_topic:=" + image_topic_
        });
        stream_manager_ = std::make_shared<StreamManager>("stream_manager", options);
        
        // 创建HTTP服务器
        http_server_ = std::make_shared<HttpServer>(io_context_, port_);
        
        // 启动工作线程
        start_workers();
        
        // 设置帧回调
        setup_frame_callback();
        
        RCLCPP_INFO(this->get_logger(), "Video Stream Server initialized:");
        RCLCPP_INFO(this->get_logger(), "  Image topic: %s", image_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Port: %d", port_);
        RCLCPP_INFO(this->get_logger(), "  Thread count: %d", thread_count_);
        RCLCPP_INFO(this->get_logger(), "  Frame rate: %d", frame_rate_);
    }
    
    ~VideoStreamServer() {
        stop_workers();
    }
    
private:
    void setup_frame_callback() {
        // 使用定时器触发回调函数，检查新帧并更新HTTP服务器
        frame_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / frame_rate_),
            [this]() {
                if (stream_manager_->has_frame()) {
                    auto frame = stream_manager_->get_current_frame();
                    http_server_->set_current_frame(frame);
                }
            });
    }
    
    void start_workers() {
        http_server_->start();
        
        // 启动IO上下文线程
        for (int i = 0; i < thread_count_; ++i) {
            workers_.emplace_back([this]() {
                io_context_.run();
            });
        }
        
        // 启动流管理器线程
        stream_thread_ = std::thread([this]() {
            rclcpp::spin(stream_manager_);
        });
    }
    
    void stop_workers() {
        // 先停止HTTP服务器
        if (http_server_) {
            http_server_->stop();
        }
        
        // 停止IO上下文
        io_context_.stop();
        
        // 等待工作线程结束
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        // 停止流管理器
        if (stream_manager_) {
            rclcpp::shutdown();  // 只影响stream_manager_
        }
        
        // 等待流管理器线程结束
        if (stream_thread_.joinable()) {
            stream_thread_.join();
        }
    }
    
    asio::io_context io_context_;
    std::shared_ptr<HttpServer> http_server_;
    std::shared_ptr<StreamManager> stream_manager_;
    
    std::vector<std::thread> workers_;
    std::thread stream_thread_;
    rclcpp::TimerBase::SharedPtr frame_timer_;
    
    std::string image_topic_;
    int port_;
    int frame_rate_;
    int jpeg_quality_;
    int thread_count_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto server = std::make_shared<VideoStreamServer>();
        rclcpp::spin(server);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Server exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}