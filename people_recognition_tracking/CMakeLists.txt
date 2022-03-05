cmake_minimum_required(VERSION 3.0.2)
project(people_recognition_tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create a symlink to compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
          ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  RESULT_VARIABLE symlinkresult
)

if (${symlinkresult} EQUAL "0")
  message(STATUS "Created symlink to compile_commands.json")
else()
  message(STATUS "Could not create symlink to compile_commands.json")
endif()

find_package(catkin REQUIRED
  COMPONENTS
  actionlib
  people_recognition_3d
  people_recognition_msgs
  roscpp
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS actionlib people_recognition_3d people_recognition_msgs roscpp
)

# ------------------------------------------------------------------------
#                         Create targets
# ------------------------------------------------------------------------

# Core library
add_library(${PROJECT_NAME}
  src/PeopleRecognitionTrackingRos.cpp
  src/Tracking.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

# ROS node
add_executable(${PROJECT_NAME}_node
  src/node.cpp
)
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)
target_include_directories(${PROJECT_NAME}_node
  PRIVATE
  ${catkin_INCLUDE_DIRS}
)

# ------------------------------------------------------------------------
#                         Install rules
# ------------------------------------------------------------------------

# Install (shared) libraries
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

# Install executables
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install header files
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# Install other common directories
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
