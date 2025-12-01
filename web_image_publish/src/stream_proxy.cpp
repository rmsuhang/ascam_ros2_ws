#include "web_image_publish/stream_manager.hpp"
#include "web_image_publish/http_client.hpp"
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <thread>
#include <vector>
#include <asio.hpp>

class StreamProxy : public rclcpp::Node {
public:
    StreamProxy() : Node("stream_proxy") {
        // 声明参数
        this->declare_parameter("source_topic", "/ascamera_hp60c/camera_publisher/rgb0/image");
        this->declare_parameter("local_port", 8081);
        this->declare_parameter("target_servers", std::vector<std::string>{});
        
        // 获取参数
        source_topic_ = this->get_parameter("source_topic").as_string();
        local_port_ = this->get_parameter("local_port").as_int();
        target_servers_ = this->get_parameter("target_servers").as_string_array();
        
        // 创建流管理器
        stream_manager_ = std::make_shared<StreamManager>("proxy_stream_manager", 
            rclcpp::NodeOptions().arguments(
                std::vector<std::string>{"--ros-args", "-r", 
                "__node:=proxy_stream_manager", "-p", "image_topic:=" + source_topic_}));
        
        // 创建HTTP客户端
        http_client_ = std::make_shared<HttpClient>(io_context_);
        
        // 启动工作线程
        start_workers();
        
        // 设置帧转发回调
        setup_forward_callback();
        
        RCLCPP_INFO(this->get_logger(), "Stream Proxy initialized:");
        RCLCPP_INFO(this->get_logger(), "  Source topic: %s", source_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Local port: %d", local_port_);
        RCLCPP_INFO(this->get_logger(), "  Target servers: %zu", target_servers_.size());
        
        for (const auto& server : target_servers_) {
            RCLCPP_INFO(this->get_logger(), "    - %s", server.c_str());
        }
    }
    
    ~StreamProxy() {
        stop_workers();
    }
    
private:
    void setup_forward_callback() {
        // 使用定时器转发帧到目标服务器
        forward_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), // ~30 FPS
            [this]() {
                if (stream_manager_->has_frame()) {
                    auto frame = stream_manager_->get_current_frame();
                    forward_to_servers(frame);
                }
            });
    }
    
    void forward_to_servers(const cv::Mat& frame) {
        for (const auto& server : target_servers_) {
            // 解析服务器地址 (格式: "host:port")
            size_t colon_pos = server.find(':');
            if (colon_pos == std::string::npos) {
                RCLCPP_WARN(this->get_logger(), "Invalid server format: %s", server.c_str());
                continue;
            }
            
            std::string host = server.substr(0, colon_pos);
            unsigned short port = std::stoi(server.substr(colon_pos + 1));
            
            // 在新线程中发送帧
            std::thread([this, host, port, frame]() {
                http_client_->send_frame(host, port, frame, 80);
            }).detach();
        }
    }
    
    void start_workers() {
        // 启动IO上下文线程
        io_thread_ = std::thread([this]() {
            io_context_.run();
        });
        
        // 启动流管理器线程
        stream_thread_ = std::thread([this]() {
            rclcpp::spin(stream_manager_);
        });
    }
    
    void stop_workers() {
        io_context_.stop();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
        
        rclcpp::shutdown();
        if (stream_thread_.joinable()) {
            stream_thread_.join();
        }
    }
    
    asio::io_context io_context_;
    std::shared_ptr<StreamManager> stream_manager_;
    std::shared_ptr<HttpClient> http_client_;
    
    std::thread io_thread_;
    std::thread stream_thread_;
    rclcpp::TimerBase::SharedPtr forward_timer_;
    
    std::string source_topic_;
    int local_port_;
    std::vector<std::string> target_servers_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto proxy = std::make_shared<StreamProxy>();
        rclcpp::spin(proxy);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Proxy exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}