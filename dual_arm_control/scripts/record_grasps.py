#!/usr/bin/env python3

import rospy
import numpy as np
import os
from geometry_msgs.msg import PoseArray, Pose

class GraspRecorder:
    def __init__(self):
        # Initialize the node. The namespace will be set externally by the launch file's <group> tag.
        # This node will now look up its own namespace using rospy.get_namespace()
        rospy.init_node('grasp_recorder')
        
        # Get the namespace the node resolved to for logging purposes
        self.node_namespace = rospy.get_namespace().strip('/')
        if not self.node_namespace:
             self.node_namespace = "global (no explicit group namespace)"

        rospy.loginfo(f"Initialized node 'grasp_recorder' in effective namespace: /{self.node_namespace}")

        self.recorded_grasps = []
        
        # ROS parameters remain private to the node (e.g., /<namespace>/grasp_recorder/~)
        self.num_poses_to_save = rospy.get_param('~num_poses_to_save', 20) 
        self.save_directory = rospy.get_param('~save_directory', os.path.join(os.path.expanduser("~"), "recorded_grasps"))
        
        # We can dynamically name the file based on the detected namespace
        default_filename = f"{self.node_namespace}_grasp_poses.npy" if self.node_namespace != "global (no explicit group namespace)" else "inferred_grasp_poses.npy"
        self.filename = rospy.get_param('~filename', default_filename)

        # Create directory if it doesn't exist
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            rospy.loginfo(f"Created save directory: {self.save_directory}")

        # Subscribe to the topic using a RELATIVE name. 
        # Since the node is inside a <group ns="X">, this topic name resolves to /X/inferred_grasp_poses
        rospy.Subscriber("inferred_grasp_poses", PoseArray, self.grasp_callback)
        
        # The fully resolved topic name will be printed here
        rospy.loginfo(f"Subscribing to resolved topic: {rospy.resolve_name('inferred_grasp_poses')}")
        rospy.loginfo(f"Will save up to {self.num_poses_to_save} latest poses to {os.path.join(self.save_directory, self.filename)}")

        rospy.on_shutdown(self.save_data)

    def grasp_callback(self, msg):
        """
        Callback function for receiving PoseArray messages.
        Stores poses up to num_poses_to_save.
        """
        if not self.recorded_grasps:
            rospy.loginfo("Received first grasp message.")
            
        # Clear previous grasps if we only want the latest set
        self.recorded_grasps = []

        for pose in msg.poses:
            # Convert Pose object to a flat list [x, y, z, qx, qy, qz, qw]
            pose_list = [
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ]
            self.recorded_grasps.append(pose_list)
        
        rospy.loginfo(f"Updated recorded grasps with {len(msg.poses)} poses. Total stored: {len(self.recorded_grasps)}")


    def save_data(self):
        """
        Saves the recorded grasps to a .npy file on shutdown.
        """
        if not self.recorded_grasps:
            rospy.logwarn("No grasps were recorded. Not saving an empty file.")
            return

        # Take only the latest 'num_poses_to_save' poses
        poses_to_save = np.array(self.recorded_grasps[-self.num_poses_to_save:])
        
        full_path = os.path.join(self.save_directory, self.filename)
        np.save(full_path, poses_to_save)
        rospy.loginfo(f"Saved {len(poses_to_save)} latest inferred grasp poses to {full_path}")

if __name__ == '__main__':
    try:
        recorder = GraspRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass