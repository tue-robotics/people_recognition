cmake_minimum_required(VERSION 3.0.2)
project(people_tracking_v2)


find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/face_recognition_node.py
  scripts/pose_estimation_node.py
  scripts/yolo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin_lint_cmake REQUIRED)
  catkin_add_catkin_lint_test()
endif()