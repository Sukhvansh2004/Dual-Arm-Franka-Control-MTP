#!/usr/bin/env python

import rospy
import actionlib
import tf2_ros
import tf.transformations as tft
import numpy as np
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3

# Import both Get and Set BodyState services and messages
from mujoco_ros_msgs.srv import SetBodyState, SetBodyStateRequest, GetBodyState, GetBodyStateRequest
from mujoco_ros_msgs.msg import BodyState

class GripperController:
    """
    A helper class to simplify interacting with different gripper types.
    - For parallel grippers, it uses standard actions.
    - For suction grippers:
        - In SIMULATION, it teleports the object to the gripper (kinematic grasp).
        - In REAL hardware, it publishes to a standard command topic.
    """
    def __init__(self, arm_id, use_suction=False):
        self.arm_id = arm_id
        self.is_suction = use_suction
        self.is_simulation = rospy.get_param("/is_simulation", False)
        
        # State for kinematic grasping
        self.attached_object = None
        self.update_timer = None
        self.T_tcp_object = None

        if self.is_suction:
            if self.is_simulation:
                rospy.loginfo(f"Initializing SIMULATED SUCTION for '{self.arm_id}' via MuJoCo services.")
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
                
                set_service_name = '/mujoco_server/set_body_state'
                rospy.wait_for_service(set_service_name, timeout=5)
                self.set_body_state_service = rospy.ServiceProxy(set_service_name, SetBodyState)
                
                get_service_name = '/mujoco_server/get_body_state'
                rospy.wait_for_service(get_service_name, timeout=5)
                self.get_body_state_service = rospy.ServiceProxy(get_service_name, GetBodyState)
                
                rospy.loginfo("MuJoCo Get/Set BodyState services found.")
            else:
                rospy.loginfo(f"Initializing REAL SUCTION gripper for arm '{self.arm_id}' via ROS topic.")
                topic = f'/{self.arm_id}/franka_suction/command'
                self.suction_pub = rospy.Publisher(topic, Bool, queue_size=1)
                rospy.sleep(1.0)
        else:
            rospy.loginfo(f"Initializing PARALLEL gripper for arm '{self.arm_id}'.")
            self.grasp_client = actionlib.SimpleActionClient(f'/{self.arm_id}/franka_gripper/grasp', GraspAction)
            self.move_client = actionlib.SimpleActionClient(f'/{self.arm_id}/franka_gripper/move', MoveAction)
            rospy.loginfo(f"Waiting for gripper action servers for arm '{self.arm_id}'...")
            self.grasp_client.wait_for_server()
            self.move_client.wait_for_server()
            rospy.loginfo(f"Gripper action servers for arm '{self.arm_id}' found.")
            
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        """Cleanly shut down the controller's resources (like the timer)."""
        rospy.loginfo(f"Shutting down GripperController for '{self.arm_id}'.")
        if self.update_timer is not None:
            self.update_timer.shutdown()
            rospy.loginfo("Kinematic grasp timer stopped.")

    def acquire(self, **kwargs):
        if self.is_suction:
            object_name = kwargs.get('object_name')
            if self.is_simulation and not object_name:
                rospy.logerr("Must provide 'object_name' for suction grasp in simulation!")
                return False
            return self._suck(object_name)
        else:
            return self._grasp(**kwargs)

    def release(self, **kwargs):
        if self.is_suction:
            return self._release_suck()
        else:
            return self._open(**kwargs)

    def _suck(self, object_name):
        if self.is_simulation:
            if self.update_timer:
                self.update_timer.shutdown()
            
            self.attached_object = object_name
            rospy.loginfo(f"Attaching '{self.attached_object}' to gripper '{self.arm_id}'.")
            
            try:
                tcp_frame = f"{self.arm_id}_cobot_pump_tcp"
                trans_world_tcp = self.tf_buffer.lookup_transform('world', tcp_frame, rospy.Time(0), rospy.Duration(1.0))
                T_world_tcp = self.stamped_transform_to_matrix(trans_world_tcp)
                
                rospy.loginfo(f"Querying MuJoCo for pose of '{self.attached_object}'...")
                get_req = GetBodyStateRequest(name=self.attached_object)
                get_res = self.get_body_state_service(get_req)
                if not get_res.success:
                    rospy.logerr(f"Failed to get pose for body '{self.attached_object}': {get_res.status_message}")
                    return False
                T_world_object = self.pose_stamped_to_matrix(get_res.state.pose)
                
                # Calculate the relative transform and store it
                T_tcp_world = tft.inverse_matrix(T_world_tcp)
                self.T_tcp_object = np.dot(T_tcp_world, T_world_object)
                
                rospy.loginfo("Calculated grasp offset. Starting kinematic attachment.")

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr(f"TF exception while calculating grasp offset: {e}")
                self.attached_object = None
                return False
            except rospy.ServiceException as e:
                rospy.logerr(f"Service exception while getting object pose: {e}")
                self.attached_object = None
                return False

            self._update_attached_object_pose()
            self.update_timer = rospy.Timer(rospy.Duration(0.001), self._update_attached_object_pose)
            rospy.loginfo("Object successfully attached to gripper.")
            return True
        else:
            rospy.loginfo(f"Arm '{self.arm_id}' suction gripper ACTIVATING.")
            self.suction_pub.publish(Bool(True))
            rospy.sleep(1.0)
            return True

    def _release_suck(self):
        if self.is_simulation:
            if self.update_timer:
                self.update_timer.shutdown()
                self.update_timer = None
                rospy.loginfo(f"Detached '{self.attached_object}' from gripper '{self.arm_id}'.")
                self.attached_object = None
                self.T_tcp_object = None
            return True
        else:
            rospy.loginfo(f"Arm '{self.arm_id}' suction gripper RELEASING.")
            self.suction_pub.publish(Bool(False))
            rospy.sleep(0.5)
            return True

    def _update_attached_object_pose(self, event=None):
        if not self.attached_object or self.T_tcp_object is None:
            return
        try:
            tcp_frame = f"{self.arm_id}_cobot_pump_tcp"
            
            trans_world_tcp = self.tf_buffer.lookup_transform('world', tcp_frame, rospy.Time(0))
            T_world_tcp = self.stamped_transform_to_matrix(trans_world_tcp)

            T_world_object = np.dot(T_world_tcp, self.T_tcp_object)
            
            pos = tft.translation_from_matrix(T_world_tcp)
            quat = tft.quaternion_from_matrix(T_world_object)

            req = SetBodyStateRequest()
            body_state = BodyState()
            body_state.name = self.attached_object
            body_state.pose.header.frame_id = 'world'
            body_state.pose.header.stamp = rospy.Time.now()
            body_state.pose.pose = Pose(position=Point(*pos), orientation=Quaternion(*quat))
            body_state.twist.header.frame_id = 'world'
            body_state.twist.header.stamp = rospy.Time.now()
            body_state.twist.twist = Twist(linear=Vector3(0,0,0), angular=Vector3(0,0,0))
            
            req.state = body_state
            req.set_pose = True
            req.set_twist = True
            
            self.set_body_state_service(req)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            if not rospy.is_shutdown():
                 rospy.logwarn(f"TF exception while updating attached object pose: {e}")

    def _grasp(self, width=0.0, force=10.0, speed=0.1, object_name=None):
        rospy.loginfo(f"Arm '{self.arm_id}' parallel gripper closing.")
        goal = GraspGoal(width=width, force=force, speed=speed)
        goal.epsilon.inner, goal.epsilon.outer = 0.05, 0.05
        self.grasp_client.send_goal(goal)
        return self.grasp_client.wait_for_result(rospy.Duration(5.0))

    def _open(self, speed=0.1, object_name=None):
        rospy.loginfo(f"Arm '{self.arm_id}' parallel gripper opening.")
        goal = MoveGoal(width=0.08, speed=speed)
        self.move_client.send_goal(goal)
        return self.move_client.wait_for_result(rospy.Duration(5.0))
        
    @staticmethod
    def stamped_transform_to_matrix(stamped_transform):
        translation = [
            stamped_transform.transform.translation.x,
            stamped_transform.transform.translation.y,
            stamped_transform.transform.translation.z,
        ]
        rotation_quat = [
            stamped_transform.transform.rotation.x,
            stamped_transform.transform.rotation.y,
            stamped_transform.transform.rotation.z,
            stamped_transform.transform.rotation.w,
        ]
        matrix = tft.quaternion_matrix(rotation_quat)
        matrix[:3, 3] = translation
        return matrix
        
    @staticmethod
    def pose_stamped_to_matrix(pose_stamped):
        translation = [
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            pose_stamped.pose.position.z,
        ]
        rotation_quat = [
            pose_stamped.pose.orientation.x,
            pose_stamped.pose.orientation.y,
            pose_stamped.pose.orientation.z,
            pose_stamped.pose.orientation.w,
        ]
        matrix = tft.quaternion_matrix(rotation_quat)
        matrix[:3, 3] = translation
        return matrix

