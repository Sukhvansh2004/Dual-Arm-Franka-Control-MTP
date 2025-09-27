#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
import tf2_ros
import tf2_geometry_msgs  # This is the package that helps with transforming poses

# Import the controller classes from our module
from dual_arm_control import ArmController, GripperController

# MODIFICATION: A new helper class to handle all coordinate transformations
class PoseTransformer:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("PoseTransformer initialized. Waiting for transforms...")
        rospy.sleep(2.0) # Give the buffer time to fill

    def transform_pose(self, pose_stamped, target_frame):
        """Transforms a PoseStamped message to the target frame."""
        try:
            transformed_pose = self.tf_buffer.transform(pose_stamped, target_frame, rospy.Duration(1))
            return transformed_pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to transform pose: {}".format(e))
            return None

def create_pose(x, y, z, roll, pitch, yaw, frame_id="world"):
    """Helper function to create a PoseStamped message from Euler angles."""
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    
    q = quaternion_from_euler(roll, pitch, yaw)
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    return pose

def main_sequence():
    """The main orchestration logic for the pick, handoff, and place task."""
    rospy.init_node("handover_sequence_controller")
    
    # MODIFICATION: Initialize our new pose transformer
    transformer = PoseTransformer()

    # Get object position from parameters
    obj_x = rospy.get_param("~object_pos_x", 0.0)
    obj_y = rospy.get_param("~object_pos_y", 0.4)

    # --- Initialization ---
    left_arm = ArmController('L_panda')
    right_arm = ArmController('R_panda')
    left_gripper = GripperController('L_panda')
    right_gripper = GripperController('R_panda')
    
    rospy.loginfo("All controllers initialized. Starting handover sequence.")
    rospy.sleep(1.0)

    # --- Define Key Poses IN THE WORLD FRAME ---
    PI = 3.14159
    # MODIFICATION: Corrected ready poses to be symmetric in the world
    ready_pose_left_world = create_pose(-0.3, 0.3, 0.5, 0, PI/2, 0)
    ready_pose_right_world = create_pose(0.3, 0.3, 0.5, -PI/2, 0, PI/2)

    pick_orientation_roll = 0
    pick_orientation_pitch = PI/2
    pick_orientation_yaw = 0
    
    handoff_orientation_roll = -PI / 2
    handoff_orientation_pitch = 0
    handoff_orientation_yaw = PI/2

    place_orientation_roll = 0
    place_orientation_pitch = PI/2
    place_orientation_yaw = PI

    pre_grasp_pose_world = create_pose(obj_x - 0.1, obj_y, 0.006, pick_orientation_roll, pick_orientation_pitch, pick_orientation_yaw)
    grasp_pose_world = create_pose(obj_x + 0.01, obj_y, 0.006, pick_orientation_roll, pick_orientation_pitch, pick_orientation_yaw)
    post_grasp_pose_world = create_pose(obj_x, obj_y, 0.3, pick_orientation_roll, pick_orientation_pitch, pick_orientation_yaw)

    pre_handoff_pose_world = create_pose(obj_x + 0.1, obj_y, 0.28, handoff_orientation_roll, handoff_orientation_pitch, handoff_orientation_yaw)
    handoff_pose_world = create_pose(obj_x, obj_y, 0.28, handoff_orientation_roll, handoff_orientation_pitch, handoff_orientation_yaw)
    place_x = -obj_x
    place_y = -obj_y
    pre_place_pose_world = create_pose(place_x, place_y, 0.3, place_orientation_roll, place_orientation_pitch, place_orientation_yaw)
    place_pose_world = create_pose(place_x, place_y, 0.125, pick_orientation_roll, pick_orientation_pitch, pick_orientation_yaw)

    # --- Start Sequence ---
    try:
        # 1. Go to Ready Poses
        rospy.loginfo("STAGE 1: Moving both arms to ready poses.")

        left_goal = transformer.transform_pose(ready_pose_left_world, 'L_panda_link0')
        right_goal = transformer.transform_pose(ready_pose_right_world, 'R_panda_link0')
        if not left_goal or not right_goal: raise Exception("TF transform failed")
        
        while not left_arm.move_to_pose(left_goal, timeout=10.0): rospy.logwarn("Retrying left arm ready pose...")
        while not right_arm.move_to_pose(right_goal, timeout=10.0): rospy.logwarn("Retrying right arm ready pose...")
        
        # 2. Left Arm Picks Object
        rospy.loginfo("STAGE 2: Left arm picking the object.")
        while not left_gripper.open(): rospy.logwarn("Retrying left gripper open...")
        
        left_goal = transformer.transform_pose(pre_grasp_pose_world, 'L_panda_link0')
        if not left_goal: raise Exception("TF transform failed")
        while not left_arm.move_to_pose(left_goal, timeout=10.0): rospy.logwarn("Retrying left arm pre-grasp pose...")
        
        left_goal = transformer.transform_pose(grasp_pose_world, 'L_panda_link0')
        if not left_goal: raise Exception("TF transform failed")
        while not left_arm.move_to_pose(left_goal, timeout=10.0): rospy.logwarn("Retrying left arm grasp pose...")
        
        while not left_gripper.grasp(width=0): rospy.logwarn("Retrying left gripper grasp...")
        rospy.sleep(1.0)
        
        left_goal = transformer.transform_pose(post_grasp_pose_world, 'L_panda_link0')
        if not left_goal: raise Exception("TF transform failed")
        while not left_arm.move_to_pose(left_goal, timeout=10.0): rospy.logwarn("Retrying left arm retract pose...")
        
        # 3. Handoff
        rospy.loginfo("STAGE 3: Handoff from left arm to right arm.")

        while not right_gripper.open(): rospy.logwarn("Retrying right gripper opening...")

        right_goal = transformer.transform_pose(pre_handoff_pose_world, 'R_panda_link0')
        if not right_goal: raise Exception("TF transform failed")
        while not right_arm.move_to_pose(right_goal, timeout=10.0): rospy.logwarn("Retrying right arm pre-handoff pose...")

        right_goal = transformer.transform_pose(handoff_pose_world, 'R_panda_link0')
        if not right_goal: raise Exception("TF transform failed")

        while not right_arm.move_to_pose(right_goal, timeout=10.0): rospy.logwarn("Retrying right arm handoff pose...")

        while not right_gripper.grasp(width=0, force=60): rospy.logwarn("Retrying right gripper grasp...")
        rospy.sleep(1.0)

        while not left_gripper.open(): rospy.logwarn("Retrying left gripper release...")

        # 4. Retract and Right Arm Places Object
        rospy.loginfo("STAGE 4: Right arm placing object.")
        left_goal = transformer.transform_pose(ready_pose_left_world, 'L_panda_link0')
        if not left_goal: raise Exception("TF transform failed")
        while not left_arm.move_to_pose(left_goal, timeout=10.0): rospy.logwarn("Retrying left arm retract...")
        
        right_goal = transformer.transform_pose(pre_place_pose_world, 'R_panda_link0')
        if not right_goal: raise Exception("TF transform failed")
        while not right_arm.move_to_pose(right_goal, timeout=10.0): rospy.logwarn("Retrying right arm pre-place pose...")
        
        right_goal = transformer.transform_pose(place_pose_world, 'R_panda_link0')
        if not right_goal: raise Exception("TF transform failed")
        while not right_arm.move_to_pose(right_goal, timeout=10.0): rospy.logwarn("Retrying right arm place pose...")
        
        while not right_gripper.open(): rospy.logwarn("Retrying right gripper release...")
        rospy.sleep(1.0)
        
        right_goal = transformer.transform_pose(pre_place_pose_world, 'R_panda_link0')
        if not right_goal: raise Exception("TF transform failed")
        while not right_arm.move_to_pose(right_goal, timeout=10.0): rospy.logwarn("Retrying right arm retract...")

        # 5. Return to Ready Poses
        rospy.loginfo("STAGE 5: Returning to ready poses.")
        right_goal = transformer.transform_pose(ready_pose_right_world, 'R_panda_link0')
        if not right_goal: raise Exception("TF transform failed")
        while not right_arm.move_to_pose(right_goal, timeout=10.0): rospy.logwarn("Retrying right arm ready pose...")

        rospy.loginfo("Handoff sequence completed successfully!")

    except Exception as e:
        rospy.logerr("An error occurred during the sequence: {}".format(e))

if __name__ == '__main__':
    main_sequence()
