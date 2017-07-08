cmake_minimum_required(VERSION 2.8.3)
project(people_detector)

find_package(catkin REQUIRED)

# catkin_python_setup()

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

catkin_package()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)