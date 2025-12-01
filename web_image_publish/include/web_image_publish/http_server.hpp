#pragma once

#include <asio.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>

using asio::ip::tcp;

class HttpServer {
public:
    HttpServer(asio::io_context& io_context, unsigned short port);
    ~HttpServer();
    
    void start();
    void stop();
    void set_current_frame(const cv::Mat& frame);
    size_t get_client_count() const;
    
private:
    void start_accept();
    void handle_accept(std::shared_ptr<tcp::socket> socket, 
                      const asio::error_code& error);
    
    asio::io_context& io_context_;
    tcp::acceptor acceptor_;
    unsigned short port_;
    
    cv::Mat current_frame_;
    mutable std::mutex frame_mutex_;
    std::atomic<bool> has_frame_{false};
    
    std::vector<std::shared_ptr<tcp::socket>> clients_;
    mutable std::mutex clients_mutex_;
    
    void send_mjpeg_stream(std::shared_ptr<tcp::socket> socket);
    void remove_client(std::shared_ptr<tcp::socket> socket);
};