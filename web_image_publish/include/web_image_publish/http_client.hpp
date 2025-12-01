#pragma once

#include <asio.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

using asio::ip::tcp;

class HttpClient {
public:
    HttpClient(asio::io_context& io_context);
    bool send_frame(const std::string& host, unsigned short port, 
                   const cv::Mat& frame, int quality = 80);
    
private:
    asio::io_context& io_context_;
    
    bool connect_to_server(tcp::socket& socket, 
                          const std::string& host, unsigned short port);
};