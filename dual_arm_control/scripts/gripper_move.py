#!/usr/bin/env python

import rospy
from dual_arm_control import GripperController

def test_gripper_sequence():
    """A simple test script to open and close both grippers."""
    rospy.init_node("test_grippers_node")

    rospy.loginfo("Initializing gripper controllers...")
    left_gripper = GripperController('L_panda')
    right_gripper = GripperController('R_panda')

    rospy.loginfo("Starting gripper test sequence.")
    rate = rospy.Rate(0.5) # 2 seconds per step

    # Open both
    rospy.loginfo("--- Opening both grippers ---")
    left_gripper.open(speed=0.1)
    right_gripper.open(speed=0.1)
    rate.sleep()

    # Close both to 2cm
    rospy.loginfo("--- Closing both grippers to 2cm ---")
    left_gripper.grasp(width=0.02, speed=0.05, force=5)
    right_gripper.grasp(width=0.02, speed=0.05, force=5)
    rate.sleep()

    # Open left, close right
    rospy.loginfo("--- Opening Left, Closing Right ---")
    left_gripper.open()
    right_gripper.grasp(width=0.0) # Fully close
    rate.sleep()

    # Close left, open right
    rospy.loginfo("--- Closing Left, Opening Right ---")
    left_gripper.grasp(width=0.0)
    right_gripper.open()
    rate.sleep()
    
    rospy.loginfo("Gripper test sequence finished.")

if __name__ == '__main__':
    try:
        test_gripper_sequence()
    except rospy.ROSInterruptException:
        pass
