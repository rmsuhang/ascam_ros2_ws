cmake_minimum_required(VERSION 3.8)
project(plc_communication)

# 检查编译器支持C++14
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 设置C++标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()


# 查找 ament_cmake_auto
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_library(SNAP7_LIBRARY snap7)

# 自动查找构建依赖
ament_auto_find_build_dependencies()


###########
## Build ##
###########
ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

# 链接 snap7 库
target_link_libraries(${PROJECT_NAME}
    ${SNAP7_LIBRARY}
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN plc_communication::PLCCommunicationNode
  EXECUTABLE ${PROJECT_NAME}_node
)

# 这个标志会构建GDB和valgrind可以读取的调试符号
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
install(FILES 3rdparty/libsnap7.so
  DESTINATION lib
)

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
)

