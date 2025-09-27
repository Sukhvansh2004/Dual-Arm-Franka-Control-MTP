#!/usr/bin/env python

import rospy
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
from std_msgs.msg import Bool

from mujoco_ros_msgs.srv import SetEqualityConstraintParameters, SetEqualityConstraintParametersRequest
from mujoco_ros_msgs.msg import EqualityConstraintParameters, EqualityConstraintType

class GripperController:
    """
    A helper class to simplify interacting with different gripper types.
    - For parallel grippers, it uses MoveAction for opening and GraspAction for grasping.
    - For suction grippers:
        - In SIMULATION, it directly creates/removes a 'weld' equality constraint in MuJoCo.
        - In REAL hardware, it publishes to a standard command topic.
    """
    def __init__(self, arm_id, use_suction=False):
        self.arm_id = arm_id
        self.is_suction = use_suction

        self.is_simulation = rospy.get_param("/is_simulation", False)

        if self.is_suction:
            if self.is_simulation:
                rospy.loginfo("Initializing SIMULATED SUCTION gripper for '{}' via MuJoCo service.".format(self.arm_id))

                service_name = '/mujoco_server/set_eq_constraint_parameters'
                rospy.wait_for_service(service_name, timeout=5)

                self.set_equality_service = rospy.ServiceProxy(service_name, SetEqualityConstraintParameters)
                rospy.loginfo("MuJoCo SetEqualityConstraintParameters service found.")
            else:
                rospy.loginfo("Initializing REAL SUCTION gripper for arm '{}' via ROS topic.".format(self.arm_id))
                topic = '/{}/franka_suction/command'.format(self.arm_id)
                self.suction_pub = rospy.Publisher(topic, Bool, queue_size=1)
                rospy.sleep(1.0)
        else:
            rospy.loginfo("Initializing PARALLEL gripper for arm '{}'.".format(self.arm_id))
            self.grasp_client = actionlib.SimpleActionClient(
                '/{}/franka_gripper/grasp'.format(arm_id), GraspAction
            )
            self.move_client = actionlib.SimpleActionClient(
                '/{}/franka_gripper/move'.format(arm_id), MoveAction
            )
            rospy.loginfo("Waiting for gripper action servers for arm '{}'...".format(self.arm_id))
            self.grasp_client.wait_for_server()
            self.move_client.wait_for_server()
            rospy.loginfo("Gripper action servers for arm '{}' found.".format(self.arm_id))

    def acquire(self, **kwargs):
        """ Acquires an object. """
        if self.is_suction:
            object_name = kwargs.get('object_name')
            if self.is_simulation and not object_name:
                rospy.logerr("Must provide 'object_name' for suction grasp in simulation!")
                return False
            return self._suck(object_name)
        else:
            return self._grasp(**kwargs)

    def release(self, **kwargs):
        """ Releases an object. """
        if self.is_suction:
            object_name = kwargs.get('object_name')
            if self.is_simulation and not object_name:
                rospy.logerr("Must provide 'object_name' for suction release in simulation!")
                return False
            return self._release_suck(object_name)
        else:
            return self._open(**kwargs)

    def _set_weld_constraint(self, object_name, active):
        """ Helper to call the MuJoCo service to activate/deactivate a weld constraint. """
        rospy.loginfo("Setting weld constraint for object '{}' to active={}.".format(object_name, active))
        try:
            req = SetEqualityConstraintParametersRequest()
            equality_msg = EqualityConstraintParameters()

            equality_msg.name = "weld_{}_{}".format(self.arm_id, object_name)
            # Use the constant from the imported message type
            equality_msg.type.value = EqualityConstraintType.WELD
            # Use the correct field names for the bodies
            equality_msg.element1 = "{}_cobot_pump".format(self.arm_id)
            equality_msg.element2 = object_name
            equality_msg.active = active
            
            # The service expects a list of parameters
            req.parameters.append(equality_msg)
            
            response = self.set_equality_service(req)
            if not response.success:
                # treat it as a success because the desired state is already met.
                if not active and "Could not find specified equality constraint" in response.status_message:
                    rospy.logwarn("Tried to release a weld constraint that did not exist. Considering it a success.")
                    return True
                else:
                    rospy.logerr("Failed to set weld constraint: {}".format(response.status_message))
                    return False
            return True
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))
            return False

    def _suck(self, object_name):
        """Activates suction."""
        if self.is_simulation:
            return self._set_weld_constraint(object_name, True)
        else:
            rospy.loginfo("Arm '{}' suction gripper ACTIVATING.".format(self.arm_id))
            self.suction_pub.publish(Bool(True))
            rospy.sleep(1.0)
            return True

    def _release_suck(self, object_name):
        """Deactivates suction."""
        if self.is_simulation:
            return self._set_weld_constraint(object_name, False)
        else:
            rospy.loginfo("Arm '{}' suction gripper RELEASING.".format(self.arm_id))
            self.suction_pub.publish(Bool(False))
            rospy.sleep(0.5)
            return True

    def _grasp(self, width=0.0, force=10.0, speed=0.1, object_name=None):
        """Private method to send a grasp command."""
        rospy.loginfo("Arm '{}' parallel gripper closing.".format(self.arm_id))
        goal = GraspGoal(width=width, force=force, speed=speed)
        goal.epsilon.inner = 0.05
        goal.epsilon.outer = 0.05
        self.grasp_client.send_goal(goal)
        return self.grasp_client.wait_for_result(rospy.Duration(5.0))

    def _open(self, speed=0.1, object_name=None):
        """Private method to open the parallel gripper."""
        rospy.loginfo("Arm '{}' parallel gripper opening.".format(self.arm_id))
        goal = MoveGoal(width=0.08, speed=speed)
        self.move_client.send_goal(goal)
        return self.move_client.wait_for_result(rospy.Duration(5.0))

