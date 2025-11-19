cmake_minimum_required(VERSION 3.5)
project(calibrate_point_cloud)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ament_index_cpp REQUIRED)

find_package(OpenCV 4.2.0 REQUIRED)

include_directories(include 
   ${PCL_CONVERSIONS_INCLUDE_DIRS}
   OpenCV 4.2.0 REQUIRED
   include )

add_executable(calibrate_node
  src/calibrate_point_cloud.cpp
  src/main.cpp
)

ament_target_dependencies(calibrate_node
  rclcpp
  rcl_interfaces
  sensor_msgs
  yaml_cpp_vendor
  visualization_msgs
  tf2_ros
  cv_bridge
  pcl_conversions
)

target_link_libraries(calibrate_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ament_index_cpp::ament_index_cpp 
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
  calibrate_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

# find dependencies


