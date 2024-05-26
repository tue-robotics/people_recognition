cmake_minimum_required(VERSION 3.0.2)
project(people_tracking_v2)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rospy
  std_msgs
  sensor_msgs
  image_recognition_msgs
  cv_bridge
)

catkin_python_setup()

add_message_files(
  FILES
  Detection.msg
  DetectionArray.msg
  SegmentedImages.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime roscpp rospy std_msgs sensor_msgs image_recognition_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/face_recognition_node.py
  scripts/pose_estimation_node.py
  scripts/yolo.py
  scripts/yolo_seg.py
  scripts/HoC.py
  src/kalman_filter.py
  tools/save-HoC-data.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test()
endif()
