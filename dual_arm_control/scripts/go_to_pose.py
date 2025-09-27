#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
import sys

class DualArmPoseController:
    """
    A class to control two Franka Panda arms by sending them to specific poses.
    This node publishes goal poses to the equilibrium_pose topic of the
    cartesian_impedance_example_controller.
    """
    def __init__(self):
        """
        Initializes the ROS node, creates publishers for each arm, and waits
        for them to connect.
        """
        rospy.init_node('dual_arm_pose_controller_node', anonymous=True)

        # Create publishers for the left and right arms' equilibrium pose topics
        self.left_arm_pub = rospy.Publisher(
            '/L_panda/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped,
            queue_size=10
        )
        self.right_arm_pub = rospy.Publisher(
            '/R_panda/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped,
            queue_size=10
        )

        # A short delay to allow publishers to establish connections
        rospy.sleep(1.0)
        rospy.loginfo("Dual Arm Pose Controller Initialized.")

    def send_pose_goal(self, arm, position, orientation):
        """
        Constructs and publishes a PoseStamped message to the specified arm.

        Args:
            arm (str): The arm to command, either 'left' or 'right'.
            position (list): A list of [x, y, z] coordinates.
            orientation (list): A list of [x, y, z, w] quaternion components.
        """
        goal_pose = PoseStamped()
        goal_pose.header.stamp = rospy.Time.now()
        
        # The frame_id MUST be the base link of the respective robot
        if arm.lower() == 'left':
            goal_pose.header.frame_id = 'L_panda_link0'
            publisher = self.left_arm_pub
        elif arm.lower() == 'right':
            goal_pose.header.frame_id = 'R_panda_link0'
            publisher = self.right_arm_pub
        else:
            rospy.logerr("Invalid arm specified. Use 'left' or 'right'.")
            return

        goal_pose.pose.position.x = position[0]
        goal_pose.pose.position.y = position[1]
        goal_pose.pose.position.z = position[2]

        goal_pose.pose.orientation.x = orientation[0]
        goal_pose.pose.orientation.y = orientation[1]
        goal_pose.pose.orientation.z = orientation[2]
        goal_pose.pose.orientation.w = orientation[3]

        publisher.publish(goal_pose)
        rospy.loginfo("Sent goal to {} arm.".format(arm))

if __name__ == '__main__':
    try:
        # Create an instance of our controller
        controller = DualArmPoseController()

        # Define a safe "home" or "ready" pose for the left arm
        left_arm_position = [0.8, 0.1, 0.4]
        # Identity quaternion (no rotation, end-effector points forward)
        left_arm_orientation = [0.0, 0.0, 0.0, 1.0]

        # Define a symmetrical pose for the right arm
        right_arm_position = [0.4, -0.2, 0.4]
        right_arm_orientation = [0.0, 0.0, 0.0, 1.0]

        # --- Send Commands ---
        rospy.loginfo("Sending Left arm to its ready position...")
        controller.send_pose_goal('left', left_arm_position, left_arm_orientation)

        # Wait a moment before sending the next command
        rospy.sleep(2.0)

        rospy.loginfo("Sending Right arm to its ready position...")
        controller.send_pose_goal('right', right_arm_position, right_arm_orientation)

        rospy.loginfo("Both arms have been commanded to their poses.")

    except rospy.ROSInterruptException:
        pass
