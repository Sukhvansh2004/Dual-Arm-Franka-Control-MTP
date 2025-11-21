#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
import tf2_ros
import tf2_geometry_msgs

from dual_arm_control import ArmController, GripperController

class PoseTransformer:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("PoseTransformer initialized. Waiting for transforms...")
        rospy.sleep(2.0)

    def transform_pose(self, pose_stamped, target_frame):
        try:
            transformed_pose = self.tf_buffer.transform(pose_stamped, target_frame, rospy.Duration(1))
            return transformed_pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to transform pose: {}".format(e))
            return None

def create_pose(x, y, z, roll, pitch, yaw, frame_id="world"):
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
    rospy.init_node("handover_sequence_controller")
    
    transformer = PoseTransformer()

    obj_x = rospy.get_param("~object_pos_x", 0.0)
    obj_y = rospy.get_param("~object_pos_y", 0.4)

    object_name_in_sim = "object_to_grasp"

    use_suction_left = rospy.get_param("~use_suction_gripper_0", False)
    use_suction_right = rospy.get_param("~use_suction_gripper_1", True)

    # --- Initialization ---
    left_arm = ArmController('L_panda')
    right_arm = ArmController('R_panda')
    left_gripper = GripperController('L_panda', use_suction=use_suction_left)
    right_gripper = GripperController('R_panda', use_suction=use_suction_right)
    
    rospy.loginfo("All controllers initialized. Starting handover sequence.")
    rospy.sleep(1.0)

    # --- Define Key Poses (Unchanged) ---
    PI = 3.14159
    ready_pose_left_world = create_pose(-0.3, 0.3, 0.5, 0, PI/2, 0)
    ready_pose_right_world = create_pose(0.3, 0.3, 0.5, -PI/2, 0, PI/2)
    pick_orientation_roll, pick_orientation_pitch, pick_orientation_yaw = 0, PI/2, 0
    handoff_orientation_roll, handoff_orientation_pitch, handoff_orientation_yaw = -PI/2, 0, PI/2
    place_orientation_roll, place_orientation_pitch, place_orientation_yaw = 0, PI/2, PI
    pre_grasp_pose_world = create_pose(obj_x - 0.1, obj_y, 0.02, pick_orientation_roll, pick_orientation_pitch, pick_orientation_yaw)
    grasp_pose_world = create_pose(obj_x + 0.01, obj_y, 0.02, pick_orientation_roll, pick_orientation_pitch, pick_orientation_yaw)
    post_grasp_pose_world = create_pose(obj_x, obj_y, 0.3, pick_orientation_roll, pick_orientation_pitch, pick_orientation_yaw)
    pre_handoff_pose_world = create_pose(obj_x + 0.25, obj_y, 0.32, handoff_orientation_roll, handoff_orientation_pitch, handoff_orientation_yaw)
    handoff_pose_world = create_pose(obj_x + 0.1, obj_y, 0.32, handoff_orientation_roll, handoff_orientation_pitch, handoff_orientation_yaw)
    place_x, place_y = -obj_x, -obj_y
    place_pose_world = create_pose(place_x, place_y, 0.3, place_orientation_roll, place_orientation_pitch, place_orientation_yaw)

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
        while not left_gripper.release(object_name=object_name_in_sim): rospy.logwarn("Retrying left gripper release...")
        left_goal = transformer.transform_pose(pre_grasp_pose_world, 'L_panda_link0')
        if not left_goal: raise Exception("TF transform failed")
        while not left_arm.move_to_pose(left_goal, timeout=10.0): rospy.logwarn("Retrying left arm pre-grasp pose...")
        left_goal = transformer.transform_pose(grasp_pose_world, 'L_panda_link0')
        if not left_goal: raise Exception("TF transform failed")
        while not left_arm.move_to_pose(left_goal, timeout=10.0): rospy.logwarn("Retrying left arm grasp pose...")
        while not left_gripper.acquire(width=0.0, object_name=object_name_in_sim): rospy.logwarn("Retrying left gripper acquire...")
        rospy.sleep(1.0)
        left_goal = transformer.transform_pose(post_grasp_pose_world, 'L_panda_link0')
        if not left_goal: raise Exception("TF transform failed")
        while not left_arm.move_to_pose(left_goal, timeout=10.0): rospy.logwarn("Retrying left arm retract pose...")
        
        # 3. Handoff
        rospy.loginfo("STAGE 3: Handoff from left arm to right arm.")
        while not right_gripper.release(object_name=object_name_in_sim): rospy.logwarn("Retrying right gripper release...")
        right_goal = transformer.transform_pose(pre_handoff_pose_world, 'R_panda_link0')
        if not right_goal: raise Exception("TF transform failed")
        while not right_arm.move_to_pose(right_goal, timeout=10.0): rospy.logwarn("Retrying right arm pre-handoff pose...")
        right_goal = transformer.transform_pose(handoff_pose_world, 'R_panda_link0')
        if not right_goal: raise Exception("TF transform failed")
        while not right_arm.move_to_pose(right_goal, timeout=10.0): rospy.logwarn("Retrying right arm handoff pose...")
        while not right_gripper.acquire(width=0.0, force=60, object_name=object_name_in_sim): rospy.logwarn("Retrying right gripper acquire...")
        rospy.sleep(1.0)
        while not left_gripper.release(object_name=object_name_in_sim): rospy.logwarn("Retrying left gripper release...")

        # 4. Retract and Right Arm Places Object
        rospy.loginfo("STAGE 4: Right arm placing object.")
        left_goal = transformer.transform_pose(ready_pose_left_world, 'L_panda_link0')
        if not left_goal: raise Exception("TF transform failed")
        while not left_arm.move_to_pose(left_goal, timeout=10.0): rospy.logwarn("Retrying left arm retract...")
        right_goal = transformer.transform_pose(place_pose_world, 'R_panda_link0')
        if not right_goal: raise Exception("TF transform failed")
        while not right_arm.move_to_pose(right_goal, timeout=10.0): rospy.logwarn("Retrying right arm pre-place pose...")
        while not right_gripper.release(object_name=object_name_in_sim): rospy.logwarn("Retrying right gripper release...")
        rospy.sleep(1.0)

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
