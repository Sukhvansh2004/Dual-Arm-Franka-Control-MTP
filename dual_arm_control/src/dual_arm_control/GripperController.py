#!/usr/bin/env python

import rospy
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal

class GripperController:
    """
    A helper class to simplify interacting with the franka_gripper action servers.
    Now uses MoveAction for opening and GraspAction for grasping.
    """
    def __init__(self, arm_id):
        self.arm_id = arm_id
        self.grasp_client = actionlib.SimpleActionClient(
            '/{}/franka_gripper/grasp'.format(arm_id),
            GraspAction
        )
        self.move_client = actionlib.SimpleActionClient(
            '/{}/franka_gripper/move'.format(arm_id),
            MoveAction
        )
        rospy.loginfo("Waiting for gripper action servers for arm '{}'...".format(self.arm_id))
        self.grasp_client.wait_for_server()
        self.move_client.wait_for_server()
        rospy.loginfo("Gripper action servers for arm '{}' found.".format(self.arm_id))

    def grasp(self, width, force=10.0, speed=0.1):
        """Sends a grasp command."""
        rospy.loginfo("Arm '{}' gripper closing.".format(self.arm_id))
        goal = GraspGoal()
        goal.width = width
        goal.epsilon.inner = 0.05
        goal.epsilon.outer = 0.05
        goal.speed = speed
        goal.force = force
        self.grasp_client.send_goal(goal)
        return self.grasp_client.wait_for_result(rospy.Duration(5.0))

    def open(self, speed=0.1):
        """
        Opens the gripper by sending a MoveGoal with a large width.
        This is the correct way to fully open the gripper.
        """
        rospy.loginfo("Arm '{}' gripper opening.".format(self.arm_id))
        goal = MoveGoal()
        goal.width = 0.08  # 8cm is fully open
        goal.speed = speed
        self.move_client.send_goal(goal)
        return self.move_client.wait_for_result(rospy.Duration(5.0))
