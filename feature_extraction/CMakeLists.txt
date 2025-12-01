cmake_minimum_required(VERSION 3.8)
project(feature_extraction)

# 检查编译器支持C++14
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  # 启用OpenMP支持（如果可用）
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel processing")
    add_compile_options(${OpenMP_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

# 设置C++标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# 查找 ament_cmake_auto
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(Eigen3 REQUIRED)  # 添加Eigen3依赖

# 自动查找构建依赖
ament_auto_find_build_dependencies()


###########
## Build ##
###########
ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)


# 链接OpenMP（如果可用）
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN feature_extraction::FeatureExtractionNode
  EXECUTABLE ${PROJECT_NAME}_node
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
ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
)

