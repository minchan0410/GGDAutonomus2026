cmake_minimum_required(VERSION 3.0.2)
project(object_detector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  vision_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  image_transport
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/object_detection.py
  scripts/object_projection.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
