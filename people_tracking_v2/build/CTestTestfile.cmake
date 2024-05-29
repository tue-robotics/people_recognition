# CMake generated Testfile for 
# Source directory: /home/miguel/ros/noetic/system/src/people_tracking_v2
# Build directory: /home/miguel/ros/noetic/system/src/people_tracking_v2/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_people_tracking_v2_catkin_lint_lint "/home/miguel/ros/noetic/system/src/people_tracking_v2/build/catkin_generated/env_cached.sh" "/home/miguel/ros/noetic/.venv/ros-noetic/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/miguel/ros/noetic/system/src/people_tracking_v2/build/test_results/people_tracking_v2/catkin_lint.xml" "--working-dir" "/home/miguel/ros/noetic/system/src/people_tracking_v2" "--return-code" "/home/miguel/ros/noetic/system/src/catkin_lint_cmake/scripts/catkin_lint_wrapper -q  --output xml /home/miguel/ros/noetic/system/src/people_tracking_v2 --output-file /home/miguel/ros/noetic/system/src/people_tracking_v2/build/test_results/people_tracking_v2/catkin_lint.xml")
set_tests_properties(_ctest_people_tracking_v2_catkin_lint_lint PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/home/miguel/ros/noetic/system/devel/share/catkin_lint_cmake/cmake/catkin_lint_cmake-extras.cmake;24;catkin_run_tests_target;/home/miguel/ros/noetic/system/src/people_tracking_v2/CMakeLists.txt;53;catkin_add_catkin_lint_test;/home/miguel/ros/noetic/system/src/people_tracking_v2/CMakeLists.txt;0;")
subdirs("gtest")
