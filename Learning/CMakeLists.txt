cmake_minimum_required(VERSION 3.0.2)
project(imitation_learning)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  ackermann_msgs
  nav_msgs
  rospy
  roscpp
  sensor_msgs
  std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()





###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  LIBRARIES lab3
 CATKIN_DEPENDS ackermann_msgs nav_msgs roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

catkin_install_python(PROGRAMS src/node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(
#   include
#   ${catkin_INCLUDE_DIRS}
# )