## overview
    ┌─────────────────────────────────┐
    │          StreamProxy            │
    │                                 │
    │  ┌─────────────────────────┐    │
    │  │     HttpClient          │    │  ← 只有一个HttpClient实例
    │  │     (单例)              │    │
    │  └─────────────────────────┘    │
    │                                 │
    │  ┌───────┐  ┌───────┐  ┌───────┐│
    │  │连接1  │  │连接2  │  │连接3  ││  ← 但可以维护多个连接
    │  └───────┘  └───────┘  └───────┘│
    └─────────────────────────────────┘
            ↓           ↓          ↓
    目标服务器A  目标服务器B  目标服务器C
## 1. 项目结构和文件功能
    1.1 package.xml

        功能: 定义ROS2包的元数据，包括包名、版本、描述、维护者、许可证以及依赖项。

        依赖项: 包括rclcpp（ROS2 C++客户端库）、sensor_msgs（传感器消息类型）、cv_bridge（OpenCV和ROS图像转换）、image_transport（图像传输）、std_msgs（标准消息类型）以及构建和运行所需的opencv、boost和asio。

    1.2 CMakeLists.txt

        功能: 构建系统的配置文件，用于编译可执行文件、链接库和安装目标。

        主要任务:

            查找依赖包（如rclcpp、OpenCV等）。

            创建两个可执行文件：video_stream_server和stream_proxy。

            指定头文件目录、链接库和安装规则。

    1.3 头文件（include/video_stream_server/）
        1.3.1 http_server.hpp

            功能: 声明HttpServer类，用于处理HTTP连接和MJPEG流传输。

            主要成员:

                io_context_: ASIO IO上下文，用于异步操作。

                acceptor_: 接受TCP连接的接收器。

                current_frame_: 存储当前视频帧。

                clients_: 存储当前连接的客户端socket列表。

                方法：启动/停止服务器、接受连接、处理客户端、发送MJPEG流等。

        1.3.2 stream_manager.hpp

            功能: 声明StreamManager类，用于管理ROS2图像订阅和帧存储。

            主要成员:

                image_sub_: 图像订阅器。

                current_frame_: 存储当前帧。

                方法：图像回调、设置/获取当前帧。

        1.3.3 http_client.hpp

            功能: 声明HttpClient类，用于向其他服务器转发视频帧。

            主要成员:

                io_context_: ASIO IO上下文。

                方法：发送帧到指定服务器、连接服务器。

    1.4 源文件（src/）
        1.4.1 video_stream_server.cpp

            功能: 主视频流服务器节点，组合StreamManager和HttpServer。

            主要工作:

                声明节点参数（图像话题、端口、帧率等）。

                创建StreamManager和HttpServer实例。

                启动工作线程（IO上下文和流管理器）。

                设置定时器，定期从StreamManager获取帧并更新HttpServer。

        1.4.2 stream_manager.cpp

            功能: 实现StreamManager类，订阅图像话题并将图像存储为OpenCV格式。

            主要工作:

                使用image_transport订阅图像话题。

                在回调中将ROS图像消息转换为OpenCV格式并存储。

        1.4.3 http_server.cpp

            功能: 实现HttpServer类，处理HTTP连接和MJPEG流。

            主要工作:

                接受客户端连接，为每个客户端创建新线程发送MJPEG流。

                将当前帧编码为JPEG，并通过HTTP multipart/x-mixed-replace流式传输。

        1.4.4 stream_proxy.cpp

            功能: 流代理节点，从源话题获取图像并转发到多个目标服务器。

            主要工作:

                使用StreamManager获取图像帧。

                使用HttpClient将帧转发到配置的目标服务器列表。

        1.4.5 http_client.cpp

            功能: 实现HttpClient类，用于向其他服务器发送视频帧。

            主要工作:

                连接到指定服务器，将帧编码为JPEG并按照MJPEG格式发送。

    1.5 启动文件（launch/）

        1.5.1 main_server.launch.py

            功能: 启动主视频流服务器节点，并设置参数。

        1.5.2 distributed_setup.launch.py

            功能: 启动分布式部署，包括一个主服务器和两个代理服务器节点。

## 2. 工作流程
    2.1 主视频流服务器

        初始化ROS2节点，声明参数。

        创建StreamManager实例，订阅图像话题。

        创建HttpServer实例，监听指定端口。

        启动工作线程：

            StreamManager在独立线程中运行，处理图像订阅。

            HttpServer在ASIO IO上下文中运行，处理HTTP连接。

        定时器定期将StreamManager中的当前帧更新到HttpServer。

    2.2 流代理节点

        初始化ROS2节点，声明参数（包括目标服务器列表）。

        创建StreamManager实例，订阅图像话题。

        创建HttpClient实例。

        启动工作线程（类似主服务器）。

        定时器定期获取当前帧，并使用HttpClient将帧转发到所有目标服务器。

    2.3 客户端访问

        客户端（如浏览器）通过HTTP连接到主服务器或代理服务器的指定端口，接收MJPEG流。

## 3. 分布式部署

    在分布式部署中，可以有多个代理节点，每个代理节点将视频流转发到多个目标服务器。这样可以实现负载均衡和冗余。