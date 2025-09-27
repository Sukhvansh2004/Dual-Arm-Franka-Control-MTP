#!/usr/bin/env python

import rospy
from dual_arm_control import GripperController

def test_gripper_sequence():
    """
    A test script to acquire and release with both grippers,
    handling both parallel and suction types.
    """
    rospy.init_node("test_grippers_node")

    # This test is designed for simulation, so we set this parameter
    # to ensure the GripperController uses the MuJoCo service for suction.
    rospy.set_param("/is_simulation", True)

    # Determine gripper types from parameters passed by the launch file
    use_suction_left = rospy.get_param("~use_suction_gripper_0", False)
    use_suction_right = rospy.get_param("~use_suction_gripper_1", False)

    # In simulation, the suction gripper needs a target body name to create a weld.
    # We use a placeholder name here to test the service call.
    dummy_object_name = "object_to_grasp" 

    rospy.loginfo("Initializing gripper controllers...")
    left_gripper = GripperController('L_panda', use_suction=use_suction_left)
    right_gripper = GripperController('R_panda', use_suction=use_suction_right)

    rospy.loginfo("Starting gripper test sequence.")
    rate = rospy.Rate(0.5) # 2 seconds per step

    # --- Release both ---
    rospy.loginfo("--- Releasing both grippers ---")
    if left_gripper.is_suction:
        left_gripper.release(object_name=dummy_object_name)
    else:
        left_gripper.release() # For parallel gripper, this is equivalent to open()
    
    if right_gripper.is_suction:
        right_gripper.release(object_name=dummy_object_name)
    else:
        right_gripper.release()
    rate.sleep()

    # --- Acquire with both ---
    rospy.loginfo("--- Acquiring with both grippers ---")
    if left_gripper.is_suction:
        left_gripper.acquire(object_name=dummy_object_name)
    else:
        left_gripper.acquire(width=0.0) # For parallel gripper, this is grasp(fully_closed)
    
    if right_gripper.is_suction:
        right_gripper.acquire(object_name=dummy_object_name)
    else:
        right_gripper.acquire(width=0.0)
    rate.sleep()

    # --- Release both again to return to a known state ---
    rospy.loginfo("--- Releasing both grippers again ---")
    if left_gripper.is_suction:
        left_gripper.release(object_name=dummy_object_name)
    else:
        left_gripper.release()
        
    if right_gripper.is_suction:
        right_gripper.release(object_name=dummy_object_name)
    else:
        right_gripper.release()
    rate.sleep()
    
    rospy.loginfo("Gripper test sequence finished.")

if __name__ == '__main__':
    try:
        test_gripper_sequence()
    except rospy.ROSInterruptException:
        pass
