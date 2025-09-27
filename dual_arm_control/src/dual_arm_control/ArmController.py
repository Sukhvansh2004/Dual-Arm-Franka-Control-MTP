#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState
import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation

def _pose_stamped_to_matrix(pose_stamped):
    """Converts a geometry_msgs/PoseStamped to a 4x4 transformation matrix."""
    p = pose_stamped.pose.position
    q = pose_stamped.pose.orientation
    
    translation = np.array([p.x, p.y, p.z])
    rotation_matrix = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation
    return matrix

def _calculate_se3_error(target_transform, current_transform):
    """
    Calculates the error between two poses in the tangent space of SE(3).

    The error is computed as the matrix logarithm of the relative transformation
    from the current pose to the target pose. This results in a 6D "twist" error vector
    [vx, vy, vz, wx, wy, wz].

    Args:
        target_transform (np.ndarray): 4x4 matrix of the target pose.
        current_transform (np.ndarray): 4x4 matrix of the current pose.

    Returns:
        np.ndarray: A 6D twist vector representing the error.
    """
    # Calculate the relative transform: T_error = T_target * T_current_inverse
    # This transform takes the current frame to the target frame.
    error_transform = np.dot(target_transform, np.linalg.inv(current_transform))

    error_matrix = scipy.linalg.logm(error_transform)

    # Extract the linear and angular error components from the error matrix.
    # The error matrix has the form:
    # [[ 0, -wz,  wy, vx],
    #  [ wz,   0, -wx, vy],
    #  [-wy,  wx,   0, vz],
    #  [  0,   0,   0,  0]]
    linear_error = error_matrix[:3, 3]
    angular_error = np.array([error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])

    # Concatenate into a 6D twist vector
    return np.concatenate((linear_error, angular_error))


class ArmController:
    """
    A helper class to simplify sending pose goals to the Cartesian impedance
    controller and waiting for the arm to reach the goal.
    """
    def __init__(self, arm_id):
        self.arm_id = arm_id
        self.pose_publisher = rospy.Publisher(
            '/{}/cartesian_impedance_example_controller/equilibrium_pose'.format(arm_id),
            PoseStamped,
            queue_size=1
        )
        self.current_transform = None # Will store the full 4x4 transform matrix
        self.pose_subscriber = rospy.Subscriber(
            '/{}/franka_state_controller/franka_states'.format(arm_id),
            FrankaState,
            self.state_callback,
            queue_size=1,
            tcp_nodelay=True
        )
        rospy.loginfo("Waiting for pose of arm '{}'...".format(self.arm_id))
        while self.current_transform is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Arm '{}' controller initialized.".format(self.arm_id))

    def state_callback(self, msg):
        # O_T_EE is the transformation from the base frame to the end-effector frame.
        # It's stored in column-major order, so we reshape and transpose.
        self.current_transform = np.transpose(np.reshape(msg.O_T_EE, (4, 4)))

    def move_to_pose(self, pose_stamped, tolerance=0.2, timeout=15.0):
        """
        Moves the arm to a target pose and waits until it arrives.
        
        Completion is determined by checking if the norm of the 6D SE(3) error
        twist is below a certain tolerance.
        """
        rospy.loginfo("Arm '{}' moving to target pose...".format(self.arm_id))
        start_time = rospy.Time.now()
        
        target_transform = _pose_stamped_to_matrix(pose_stamped)
        
        rate = rospy.Rate(1) # Publish goal at 100 Hz
        while (rospy.Time.now() - start_time) < rospy.Duration(timeout):
            self.pose_publisher.publish(pose_stamped)
            
            if self.current_transform is not None:
                # Calculate the 6D error twist vector
                error_twist = _calculate_se3_error(target_transform, self.current_transform)
                
                # The "distance" is the norm of this twist vector
                error_norm = np.linalg.norm(error_twist)

                if error_norm < tolerance:
                    rospy.loginfo("Arm '{}' reached target pose (SE(3) error norm: {:.4f}).".format(self.arm_id, error_norm))
                    # Publish for a short duration more to ensure stability
                    for _ in range(2):
                        self.pose_publisher.publish(pose_stamped)
                        rate.sleep()
                    return True
            rate.sleep()
        
        rospy.logwarn("Arm '{}' failed to reach target pose within timeout.".format(self.arm_id))
        return False
