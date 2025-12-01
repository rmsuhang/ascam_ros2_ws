#include "web_image_publish/http_client.hpp"
#include <sstream>
/**
 * @brief HttpClient构造函数
 * @param io_context ASIO IO上下文，用于管理异步I/O操作
 * @note 使用外部传入的io_context，便于在多个客户端间共享IO资源
 */
HttpClient::HttpClient(asio::io_context& io_context)
    : io_context_(io_context) {
}

/**
 * @brief 发送单帧图像到指定的HTTP服务器
 * @param host 目标服务器主机名或IP地址
 * @param port 目标服务器端口号
 * @param frame 要发送的OpenCV图像帧
 * @param quality JPEG压缩质量(0-100)
 * @return 发送成功返回true，失败返回false
 * 
 * @details 该函数执行以下步骤：
 * 1. 创建TCP socket连接
 * 2. 将图像编码为JPEG格式
 * 3. 构建MJPEG格式的数据包
 * 4. 通过socket发送数据
 * 5. 发送完成后自动关闭连接
 * 
 * @note 这是同步阻塞操作，调用线程会等待整个发送过程完成
 */
bool HttpClient::send_frame(const std::string& host, unsigned short port, 
                           const cv::Mat& frame, int quality) {
    try {
        tcp::socket socket(io_context_);
        
        if (!connect_to_server(socket, host, port)) {
            return false;
        }
        
        // 编码为JPEG
        std::vector<uchar> jpeg_buffer;
        std::vector<int> params = {
            cv::IMWRITE_JPEG_QUALITY, quality,
            cv::IMWRITE_JPEG_OPTIMIZE, 1
        };
        
        if (!cv::imencode(".jpg", frame, jpeg_buffer, params)) {
            return false;
        }
        
        // 构建MJPEG帧
        std::stringstream frame_header;
        frame_header << "--frame\r\n"
                   << "Content-Type: image/jpeg\r\n"
                   << "Content-Length: " << jpeg_buffer.size() << "\r\n"
                   << "\r\n";
        
        // 发送帧头和数据
        std::string header_str = frame_header.str();
        std::vector<asio::const_buffer> buffers;
        buffers.push_back(asio::buffer(header_str));
        buffers.push_back(asio::buffer(jpeg_buffer));
        buffers.push_back(asio::buffer("\r\n", 2));
        
        asio::write(socket, buffers);
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to send frame to " << host << ":" << port 
                  << " - " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 连接到指定的TCP服务器
 * @param socket 要使用的socket引用
 * @param host 目标服务器主机名或IP地址
 * @param port 目标服务器端口号
 * @return 连接成功返回true，失败返回false
 * 
 * @details 使用ASIO的解析器和连接功能建立TCP连接
 */
bool HttpClient::connect_to_server(tcp::socket& socket, 
                                  const std::string& host, unsigned short port) {
    try {
        tcp::resolver resolver(io_context_);
        tcp::resolver::results_type endpoints = resolver.resolve(host, std::to_string(port));
        
        asio::connect(socket, endpoints);
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Connection failed to " << host << ":" << port 
                  << " - " << e.what() << std::endl;
        return false;
    }
}