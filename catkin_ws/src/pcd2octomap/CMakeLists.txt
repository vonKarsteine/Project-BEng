cmake_minimum_required(VERSION 3.0.2)
project(pcd2octomap)

find_package(catkin REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(octomap REQUIRED)

catkin_package()

include_directories(${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS})

add_executable(pcd2octomap src/pcd2octomap.cpp)
target_link_libraries(pcd2octomap ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})
