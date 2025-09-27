#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState
import tf.transformations as tft
import numpy as np

# For reading single key presses
import sys
import select
import tty
import termios

class GetKey:
    """
    A helper class to read single key presses from the terminal without
    requiring the user to press Enter.
    """
    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self, timeout=0.1):
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None

class DualArmTeleopController:
    """
    A class for teleoperating two Franka Panda arms simultaneously
    using keyboard commands. It reads the current robot poses and publishes
    incremental changes.
    """
    def __init__(self):
        rospy.init_node('dual_arm_teleop_keyboard_controller', anonymous=True)

        # --- Setup for Left Arm ---
        self.left_publisher = rospy.Publisher(
            '/L_panda/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped,
            queue_size=10
        )
        self.left_current_pose = None
        self.left_pose_received = False
        rospy.Subscriber(
            '/L_panda/franka_state_controller/franka_states',
            FrankaState,
            self.left_franka_state_callback
        )

        # --- Setup for Right Arm ---
        self.right_publisher = rospy.Publisher(
            '/R_panda/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped,
            queue_size=10
        )
        self.right_current_pose = None
        self.right_pose_received = False
        rospy.Subscriber(
            '/R_panda/franka_state_controller/franka_states',
            FrankaState,
            self.right_franka_state_callback
        )

        self.linear_speed = 0.05  # meters per key press
        self.angular_speed = 0.1  # radians per key press

    def process_franka_state(self, msg, arm_id):
        """Helper function to convert FrankaState message to PoseStamped."""
        initial_pose = np.transpose(np.reshape(msg.O_T_EE, (4, 4)))
        
        pose = PoseStamped()
        pose.header.frame_id = '{}_link0'.format(arm_id)
        
        pose.pose.position.x = initial_pose[0, 3]
        pose.pose.position.y = initial_pose[1, 3]
        pose.pose.position.z = initial_pose[2, 3]
        
        quat = tft.quaternion_from_matrix(initial_pose)
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        return pose

    def left_franka_state_callback(self, msg):
        """Callback to update the current pose of the left arm."""
        self.left_current_pose = self.process_franka_state(msg, 'L_panda')
        if not self.left_pose_received:
            self.left_pose_received = True

    def right_franka_state_callback(self, msg):
        """Callback to update the current pose of the right arm."""
        self.right_current_pose = self.process_franka_state(msg, 'R_panda')
        if not self.right_pose_received:
            self.right_pose_received = True

    def print_instructions(self):
        """Prints the control instructions for the user."""
        print("""
----------------------------------
Dual Arm Keyboard Teleop Controller
----------------------------------
       LEFT ARM    |    RIGHT ARM
----------------------------------
w/s : fwd/back +/-x| i/k : fwd/back +/-x
a/d : left/right +/-y| j/l : left/right +/-y
q/e : up/down +/-z   | u/o : up/down +/-z
----------------------------------
CTRL-C to quit
----------------------------------""")

    def run(self):
        """The main control loop."""
        self.print_instructions()
        
        rospy.loginfo("Waiting for robot pose messages...")
        while not (self.left_pose_received and self.right_pose_received) and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        if rospy.is_shutdown():
            return
            
        rospy.loginfo("Ready to teleoperate.")
        
        left_target_pose = self.left_current_pose
        right_target_pose = self.right_current_pose
        
        with GetKey() as getKey:
            while not rospy.is_shutdown():
                # Always start with the current pose to command relatively
                left_target_pose = self.left_current_pose
                right_target_pose = self.right_current_pose

                key = getKey.get_key()
                if key:
                    # --- Left Arm Controls ---
                    if key == 'w':
                        left_target_pose.pose.position.x += self.linear_speed
                    elif key == 's':
                        left_target_pose.pose.position.x -= self.linear_speed
                    elif key == 'a':
                        left_target_pose.pose.position.y += self.linear_speed
                    elif key == 'd':
                        left_target_pose.pose.position.y -= self.linear_speed
                    elif key == 'q':
                        left_target_pose.pose.position.z += self.linear_speed
                    elif key == 'e':
                        left_target_pose.pose.position.z -= self.linear_speed
                    
                    # --- Right Arm Controls ---
                    elif key == 'i':
                        right_target_pose.pose.position.x += self.linear_speed
                    elif key == 'k':
                        right_target_pose.pose.position.x -= self.linear_speed
                    elif key == 'j':
                        right_target_pose.pose.position.y += self.linear_speed
                    elif key == 'l':
                        right_target_pose.pose.position.y -= self.linear_speed
                    elif key == 'u':
                        right_target_pose.pose.position.z += self.linear_speed
                    elif key == 'o':
                        right_target_pose.pose.position.z -= self.linear_speed
                
                # Always publish both poses to keep the controllers active and stiff
                if left_target_pose and right_target_pose:
                    left_target_pose.header.stamp = rospy.Time.now()
                    right_target_pose.header.stamp = rospy.Time.now()
                    self.left_publisher.publish(left_target_pose)
                    self.right_publisher.publish(right_target_pose)

                rospy.sleep(0.05) # Loop at ~20Hz

if __name__ == '__main__':
    try:
        controller = DualArmTeleopController()
        controller.run()
    except rospy.ROSInterruptException:
        pass

