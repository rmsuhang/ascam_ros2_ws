#include "web_image_publish/http_server.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>

/**
 * @brief HttpServer构造函数
 * @param io_context ASIO IO上下文，用于管理所有异步I/O操作
 * @param port 服务器监听的端口号
 * 
 * @details 初始化服务器，创建TCP acceptor准备接受客户端连接
 * 注意：使用外部传入的io_context，便于在更大的应用中集成和管理
 */
HttpServer::HttpServer(asio::io_context& io_context, unsigned short port)
    : io_context_(io_context), acceptor_(io_context, tcp::endpoint(tcp::v4(), port)), 
      port_(port) {
}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::start() {
    start_accept();
}

void HttpServer::stop() {
    asio::error_code ec;
    acceptor_.close(ec);
    
    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (auto& client : clients_) {
        if (client->is_open()) {
            client->close();
        }
    }
    clients_.clear();
}

/**
 * @brief 设置当前要推送的视频帧
 * @param frame OpenCV图像帧
 * 
 * @details 更新当前帧，所有连接的客户端都会收到这个新帧
 * 使用互斥锁保护帧数据，确保线程安全，一次只能有一个线程执行这段代码，函数结束，lock_guard析构，自动调用frame_mutex_.unlock()
 */
void HttpServer::set_current_frame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    current_frame_ = frame.clone();
    has_frame_ = true;
}

size_t HttpServer::get_client_count() const {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    return clients_.size();
}

/**
 * @brief 开始异步接受客户端连接
 * 
 * @details 核心异步操作：
 * 1. 创建新的socket对象
 * 2. 启动async_accept异步等待客户端连接
 * 3. 当有客户端连接时，handle_accept回调会被调用
 */
void HttpServer::start_accept() {
    auto socket = std::make_shared<tcp::socket>(io_context_);
    
    acceptor_.async_accept(*socket,
        [this, socket](const asio::error_code& error) {
            handle_accept(socket, error);
        });
}

/**
 * @brief 处理新客户端连接
 * @param socket 客户端socket的shared_ptr
 * @param error 错误码，表示连接是否成功
 * 
 * @details 当async_accept完成时调用：
 * 1. 检查连接是否成功
 * 2. 记录客户端信息
 * 3. 将客户端添加到管理列表
 * 4. 创建新线程专门服务这个客户端
 * 5. 继续接受其他客户端连接
 */
void HttpServer::handle_accept(std::shared_ptr<tcp::socket> socket, 
                              const asio::error_code& error) {
    if (!error) {
        std::string client_ip = socket->remote_endpoint().address().to_string();
        std::cout << "New client connected from: " << client_ip << std::endl;
        
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            clients_.push_back(socket);
        }
        
        // 在新线程中处理客户端，线程会调用send_mjpeg_stream，不断推送新帧，直到客户端断开连接
        std::thread([this, socket]() {
            send_mjpeg_stream(socket);
        }).detach();// detach: 分离线程，让它在后台自主运行
    } else {
        std::cerr << "Accept error: " << error.message() << std::endl;
    }
    
    // 继续接受新连接
    start_accept();
}

/**
 * @brief 向客户端发送MJPEG视频流
 * @param socket 客户端socket的shared_ptr
 * 
 * @details 这是每个客户端线程的核心函数：
 * 1. 发送HTTP MJPEG流头部
 * 2. 进入无限循环，持续推送视频帧
 * 3. 控制帧率，确保按30FPS发送
 * 4. 处理客户端断开和错误情况
 * 
 * 线程生命周期：这个函数会阻塞在循环中，直到客户端断开连接
 */
void HttpServer::send_mjpeg_stream(std::shared_ptr<tcp::socket> socket) {
    try {
        const std::string boundary = "frame";
        const std::string header = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=" + boundary + "\r\n"
            "Connection: close\r\n"
            "Max-Age: 0\r\n"
            "Expires: 0\r\n"
            "Cache-Control: no-cache, private\r\n"
            "Pragma: no-cache\r\n"
            "\r\n";
        
        // 发送HTTP头
        asio::write(*socket, asio::buffer(header));
        
        auto last_frame_time = std::chrono::steady_clock::now();
        auto frame_interval = std::chrono::milliseconds(1000 / 30); // 30 FPS
        
        while (socket->is_open()) {
            auto current_time = std::chrono::steady_clock::now();
            if (current_time - last_frame_time < frame_interval) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            cv::Mat frame;
            bool valid_frame = false;
            
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                if (has_frame_ && !current_frame_.empty()) {
                    frame = current_frame_.clone();
                    valid_frame = true;
                }
            }
            
            if (!valid_frame) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            // 编码为JPEG
            std::vector<uchar> jpeg_buffer;
            std::vector<int> params = {
                cv::IMWRITE_JPEG_QUALITY, 80,
                cv::IMWRITE_JPEG_OPTIMIZE, 1
            };
            
            if (!cv::imencode(".jpg", frame, jpeg_buffer, params)) {
                std::cerr << "Failed to encode frame to JPEG" << std::endl;
                continue;
            }
            
            // 构建MJPEG帧
            std::stringstream frame_header;
            frame_header << "--" << boundary << "\r\n"
                       << "Content-Type: image/jpeg\r\n"
                       << "Content-Length: " << jpeg_buffer.size() << "\r\n"
                       << "\r\n";
            
            // 发送帧头和数据
            std::string header_str = frame_header.str();
            std::vector<asio::const_buffer> buffers;
            buffers.push_back(asio::buffer(header_str));
            buffers.push_back(asio::buffer(jpeg_buffer));
            buffers.push_back(asio::buffer("\r\n", 2));
            
            asio::error_code ec;
            asio::write(*socket, buffers, ec);
            
            if (ec) {
                throw asio::system_error(ec);
            }
            
            last_frame_time = current_time;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Client disconnected: " << e.what() << std::endl;
    }
    
    remove_client(socket);
}

void HttpServer::remove_client(std::shared_ptr<tcp::socket> socket) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    clients_.erase(
        std::remove(clients_.begin(), clients_.end(), socket),
        clients_.end());
}