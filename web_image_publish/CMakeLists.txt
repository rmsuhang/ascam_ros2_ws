cmake_minimum_required(VERSION 3.8)
project(web_image_publish)

# 检查编译器支持C++14
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 设置C++标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# asio_cmake_module已经安装在/opt/ros/humble/share
list(APPEND CMAKE_MODULE_PATH "/opt/ros/humble/share/asio_cmake_module/cmake/Modules")

# 查找 ament_cmake_auto
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(ASIO REQUIRED)



###########
## Build ##
###########

# 包含目录
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

# 创建可执行文件
add_executable(video_stream_server
    src/video_stream_server.cpp
    src/http_server.cpp
    src/stream_manager.cpp
)

# 明确添加依赖
ament_target_dependencies(video_stream_server
    rclcpp
    sensor_msgs
    std_msgs
    cv_bridge
    image_transport
)

# 链接库
target_link_libraries(video_stream_server
    ${OpenCV_LIBRARIES}
    pthread
)

add_executable(stream_proxy
    src/stream_proxy.cpp
    src/http_client.cpp
    src/stream_manager.cpp
)

ament_target_dependencies(stream_proxy
    rclcpp
    sensor_msgs
    std_msgs
    cv_bridge
    image_transport
)

target_link_libraries(stream_proxy
    ${OpenCV_LIBRARIES}
    pthread
)


# 构建GDB和valgrind可以读取的调试符号
add_compile_options(-g)

# 测试部分（可选）
if(BUILD_TESTING)
  # 查找测试依赖
  find_package(ament_lint_auto REQUIRED)
  
  # 自动添加lint测试
  ament_lint_auto_find_test_dependencies()
endif()


#############
## Install ##
#############
# 安装目标
install(TARGETS
    video_stream_server
    stream_proxy
    DESTINATION lib/${PROJECT_NAME}
)

# 安装头文件
install(DIRECTORY include/
    DESTINATION include
)

# 必须调用ament_package()
ament_package()