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
  HoCVector.msg
  HoCVectorArray.msg
  BodySize.msg
  BodySizeArray.msg
  ComparisonScores.msg
  ComparisonScoresArray.msg
  DecisionResult.msg
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
  scripts/yolo_seg.py
  scripts/HoC.py
  scripts/comparison_node.py
  scripts/decision_node.py
  scripts/central_node.py
  src/kalman_filter.py
  tools/recorder.py
  tools/save_HoC_data.py
  tools/save_pose_data.py
  testing/rgb_publisher.py
  testing/depth_publisher.py
  testing/frequency_measure.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test()
endif()
