cmake_minimum_required(VERSION 3.5)
project(process_point_cloud)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "$\{ORIGIN\};$\{ORIGIN\}/../lib")

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

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

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

ament_package()

find_package(PCL REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)

include_directories(include)
link_directories(3rdparty)
add_executable(process_node
  src/process_node.cpp
  src/utils.cpp
  src/snap7.cpp
  src/main.cpp
)

ament_target_dependencies(process_node
  rclcpp
  sensor_msgs
  yaml_cpp_vendor
  ament_index_cpp
  pcl_conversions
  cv_bridge
  std_msgs
  vision_msgs
  rcl_interfaces
)

target_link_libraries(process_node
  ${PCL_LIBRARIES}
  snap7
)

install(TARGETS process_node
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES 3rdparty/libsnap7.so
  DESTINATION lib
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
