cmake_minimum_required(VERSION 2.8.3)
project(people_detection_3d_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  sensor_msgs
  image_recognition_msgs
  geometry_msgs
)

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  Person3D.msg
)

# Generate services in the 'srv' folder
add_service_files(
  FILES
  DetectPeople3D.srv
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
  image_recognition_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime sensor_msgs image_recognition_msgs geometry_msgs
)

# Test catkin lint
find_program(CATKIN_LINT catkin_lint REQUIRED)
execute_process(COMMAND "${CATKIN_LINT}" "-q" "-W2" "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE lint_result)
if(NOT ${lint_result} EQUAL 0)
  message(FATAL_ERROR "catkin_lint failed")
endif()
