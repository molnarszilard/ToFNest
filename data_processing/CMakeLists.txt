cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(tofnest_data)

find_package(PCL 1.2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  geometry_msgs
  topic_tools
  visualization_msgs
  cv_bridge
  message_filters
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (normal2rgb src/normal2rgb.cpp)
add_executable (compute_normals src/compute_normals.cpp)
add_executable (cloud2image src/cloud2image.cpp)
add_executable (sweep src/sweep_nan.cpp)
add_executable (normal_performance src/normal_performance.cpp)
add_executable (depth3 src/depth3.cpp)
add_executable (depth2pcd src/depth2pcd.cpp)
add_executable (depth2pcd_color src/depth2pcd_color.cpp)
add_executable (depth2pcd_normal src/depth2pcd_normal.cpp)
add_executable (addnoise2depth src/addnoise2depth.cpp)
add_executable (save_depth_images src/save_depth_images.cpp)

target_link_libraries( normal2rgb ${PCL_LIBRARIES})
target_link_libraries( compute_normals ${PCL_LIBRARIES})
target_link_libraries( cloud2image ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries( sweep ${PCL_LIBRARIES})
target_link_libraries( normal_performance ${PCL_LIBRARIES})
target_link_libraries( depth3 ${OpenCV_LIBS})
target_link_libraries( depth2pcd ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries( depth2pcd_color ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries( depth2pcd_normal ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries( addnoise2depth ${OpenCV_LIBS})
target_link_libraries (save_depth_images ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})
