# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "people_tracking_v2: 5 messages, 0 services")

set(MSG_I_FLAGS "-Ipeople_tracking_v2:/home/miguel/ros/noetic/system/src/people_tracking_v2/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(people_tracking_v2_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg" NAME_WE)
add_custom_target(_people_tracking_v2_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "people_tracking_v2" "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg" ""
)

get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg" NAME_WE)
add_custom_target(_people_tracking_v2_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "people_tracking_v2" "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg" "people_tracking_v2/Detection"
)

get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg" NAME_WE)
add_custom_target(_people_tracking_v2_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "people_tracking_v2" "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg" "std_msgs/Header:sensor_msgs/Image"
)

get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg" NAME_WE)
add_custom_target(_people_tracking_v2_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "people_tracking_v2" "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg" NAME_WE)
add_custom_target(_people_tracking_v2_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "people_tracking_v2" "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg" "std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_cpp(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg"
  "${MSG_I_FLAGS}"
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_cpp(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_cpp(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_cpp(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/people_tracking_v2
)

### Generating Services

### Generating Module File
_generate_module_cpp(people_tracking_v2
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/people_tracking_v2
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(people_tracking_v2_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(people_tracking_v2_generate_messages people_tracking_v2_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_cpp _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_cpp _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_cpp _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_cpp _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_cpp _people_tracking_v2_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(people_tracking_v2_gencpp)
add_dependencies(people_tracking_v2_gencpp people_tracking_v2_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS people_tracking_v2_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_eus(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg"
  "${MSG_I_FLAGS}"
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_eus(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_eus(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_eus(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/people_tracking_v2
)

### Generating Services

### Generating Module File
_generate_module_eus(people_tracking_v2
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/people_tracking_v2
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(people_tracking_v2_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(people_tracking_v2_generate_messages people_tracking_v2_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_eus _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_eus _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_eus _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_eus _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_eus _people_tracking_v2_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(people_tracking_v2_geneus)
add_dependencies(people_tracking_v2_geneus people_tracking_v2_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS people_tracking_v2_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_lisp(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg"
  "${MSG_I_FLAGS}"
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_lisp(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_lisp(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_lisp(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/people_tracking_v2
)

### Generating Services

### Generating Module File
_generate_module_lisp(people_tracking_v2
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/people_tracking_v2
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(people_tracking_v2_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(people_tracking_v2_generate_messages people_tracking_v2_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_lisp _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_lisp _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_lisp _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_lisp _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_lisp _people_tracking_v2_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(people_tracking_v2_genlisp)
add_dependencies(people_tracking_v2_genlisp people_tracking_v2_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS people_tracking_v2_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_nodejs(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg"
  "${MSG_I_FLAGS}"
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_nodejs(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_nodejs(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_nodejs(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/people_tracking_v2
)

### Generating Services

### Generating Module File
_generate_module_nodejs(people_tracking_v2
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/people_tracking_v2
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(people_tracking_v2_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(people_tracking_v2_generate_messages people_tracking_v2_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_nodejs _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_nodejs _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_nodejs _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_nodejs _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_nodejs _people_tracking_v2_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(people_tracking_v2_gennodejs)
add_dependencies(people_tracking_v2_gennodejs people_tracking_v2_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS people_tracking_v2_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_py(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg"
  "${MSG_I_FLAGS}"
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_py(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_py(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2
)
_generate_msg_py(people_tracking_v2
  "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2
)

### Generating Services

### Generating Module File
_generate_module_py(people_tracking_v2
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(people_tracking_v2_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(people_tracking_v2_generate_messages people_tracking_v2_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/Detection.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_py _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/DetectionArray.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_py _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/SegmentedImages.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_py _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/HoCVector.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_py _people_tracking_v2_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/miguel/ros/noetic/system/src/people_tracking_v2/msg/BodySize.msg" NAME_WE)
add_dependencies(people_tracking_v2_generate_messages_py _people_tracking_v2_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(people_tracking_v2_genpy)
add_dependencies(people_tracking_v2_genpy people_tracking_v2_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS people_tracking_v2_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/people_tracking_v2)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/people_tracking_v2
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(people_tracking_v2_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(people_tracking_v2_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/people_tracking_v2)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/people_tracking_v2
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(people_tracking_v2_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(people_tracking_v2_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/people_tracking_v2)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/people_tracking_v2
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(people_tracking_v2_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(people_tracking_v2_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/people_tracking_v2)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/people_tracking_v2
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(people_tracking_v2_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(people_tracking_v2_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2)
  install(CODE "execute_process(COMMAND \"/home/miguel/ros/noetic/.venv/ros-noetic/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2
    DESTINATION ${genpy_INSTALL_DIR}
    # skip all init files
    PATTERN "__init__.py" EXCLUDE
    PATTERN "__init__.pyc" EXCLUDE
  )
  # install init files which are not in the root folder of the generated code
  string(REGEX REPLACE "([][+.*()^])" "\\\\\\1" ESCAPED_PATH "${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2")
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/people_tracking_v2
    DESTINATION ${genpy_INSTALL_DIR}
    FILES_MATCHING
    REGEX "${ESCAPED_PATH}/.+/__init__.pyc?$"
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(people_tracking_v2_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(people_tracking_v2_generate_messages_py sensor_msgs_generate_messages_py)
endif()
