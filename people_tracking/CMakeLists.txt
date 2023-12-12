cmake_minimum_required(VERSION 3.0.2)
project(people_tracking)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  people_recognition_msgs
  rgbd
  rgbd_image_buffer
  roscpp
  sensor_msgs
  visualization_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  ColourCheckedTarget.msg
  ColourTarget.msg
  DetectPerson.msg
  DetectedPerson.msg
)

add_service_files(
  FILES
  Depth.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime sensor_msgs
)

include_directories(
# include
 SYSTEM
 ${catkin_INCLUDE_DIRS}
)

add_executable(people_tracker src/people_tracker.cpp)
target_link_libraries(people_tracker ${catkin_LIBRARIES})

install(
  TARGETS people_tracker
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if (CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test(-W2)
endif()
