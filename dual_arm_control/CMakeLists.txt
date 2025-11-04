cmake_minimum_required(VERSION 3.0.2)
project(dual_arm_control)

## Find all necessary ROS packages for your entire project
find_package(catkin REQUIRED COMPONENTS
  rospy
  actionlib
  franka_gripper
  franka_msgs
  geometry_msgs
  message_generation
)

## This is the most important command. It tells catkin to look for a setup.py
## and use it to install your Python code as an importable module.
catkin_python_setup()

## --------------------------------------------------
# Add your .srv file here
add_service_files(
  FILES
  GetGrasps.srv
)

# Generate the Python/C++ code for your services/messages
generate_messages(
  DEPENDENCIES
  geometry_msgs  # Dependency for PoseArray
)
## --------------------------------------------------

## This macro generates cmake config files for your package.
## We list all ROS dependencies that other packages might need if they depend on this one.
catkin_package(
  CATKIN_DEPENDS rospy actionlib franka_gripper franka_msgs geometry_msgs
  message_runtime
)

## Install the executable scripts from the 'scripts' folder.
## These are the files you will run with 'rosrun' or 'roslaunch'.
catkin_install_python(PROGRAMS
  scripts/handover_sequence.py
  scripts/gripper_move.py
  scripts/go_to_pose.py
  scripts/teleop_keyboard.py
  scripts/fastsam_client.py
  scripts/fastsam_server.py
  scripts/grasp_client.py
  scripts/grasp_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


## Install your launch, worlds, and srv files so ROS can find them.
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
)
install(DIRECTORY srv/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/srv
)