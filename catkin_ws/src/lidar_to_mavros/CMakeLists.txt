cmake_minimum_required(VERSION 3.0.2)
project(lidar_to_mavros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_executable(lidar_to_mavros_node src/lidar_to_mavros.cpp)
target_link_libraries(lidar_to_mavros_node ${catkin_LIBRARIES})
