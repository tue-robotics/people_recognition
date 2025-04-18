cmake_minimum_required(VERSION 3.5)
project(people_recognition_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  sensor_msgs
  image_recognition_msgs
  geometry_msgs
)

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  Person2D.msg
  Person3D.msg
  People3D.msg
)

# Generate services in the 'srv' folder
add_service_files(
  FILES
  RecognizePeople2D.srv
  RecognizePeople3D.srv
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

if (CATKIN_ENABLE_TESTING)
    # Test catkin lint
    find_program(CATKIN_LINT catkin_lint REQUIRED)
    execute_process(COMMAND "${CATKIN_LINT}" "-q" "-W2" "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE lint_result)
    if(NOT ${lint_result} EQUAL 0)
      message(FATAL_ERROR "catkin_lint failed")
    endif()
endif()
