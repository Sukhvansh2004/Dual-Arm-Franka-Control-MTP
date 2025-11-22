#!/usr/bin/env python
import rospy
import tf
import numpy as np
import os

# Import MuJoCo services to get object state directly
from mujoco_ros_msgs.srv import GetBodyState, GetBodyStateRequest

class PoseRecorder:
    def __init__(self):
        rospy.init_node('pose_recorder', anonymous=True)

        self.l_panda_poses = []
        self.r_panda_poses = []
        self.object_poses = []
        
        # Name of the object in MuJoCo
        self.object_name = "object_to_grasp"

        self.listener = tf.TransformListener()

        # Setup MuJoCo Service to get object pose
        service_name = '/mujoco_server/get_body_state'
        rospy.loginfo(f"Waiting for service: {service_name}")
        rospy.wait_for_service(service_name)
        self.get_body_state_service = rospy.ServiceProxy(service_name, GetBodyState)
        rospy.loginfo("Service found. Recorder ready.")

        rospy.on_shutdown(self.save_data)

        self.rate = rospy.Rate(2)

    def get_object_pose(self):
        """
        Calls MuJoCo service to get [x, y, z, qx, qy, qz, qw] for the object.
        """
        try:
            req = GetBodyStateRequest(name=self.object_name)
            res = self.get_body_state_service(req)
            
            if res.success:
                p = res.state.pose.pose.position
                o = res.state.pose.pose.orientation
                # Return standard [x, y, z, qx, qy, qz, qw] format
                return [p.x, p.y, p.z, o.x, o.y, o.z, o.w]
            else:
                rospy.logwarn_throttle(2, f"Failed to get state for {self.object_name}: {res.status_message}")
                return None
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None

    def run(self):
        rospy.loginfo("Recording started...")
        while not rospy.is_shutdown():
            try:
                # 1. Get L_panda pose (TF)
                (trans_l, rot_l) = self.listener.lookupTransform('/world', '/L_panda_EE', rospy.Time(0))
                
                # 2. Get R_panda pose (TF)
                (trans_r, rot_r) = self.listener.lookupTransform('/world', '/R_panda_EE', rospy.Time(0))
                
                # 3. Get Object pose (MuJoCo Service)
                obj_pose = self.get_object_pose()

                # Only append if we successfully got data for all three
                if obj_pose is not None:
                    self.l_panda_poses.append(trans_l + rot_l)
                    self.r_panda_poses.append(trans_r + rot_r)
                    self.object_poses.append(obj_pose)
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            self.rate.sleep()

    def save_data(self):
        rospy.loginfo("Saving recorded poses...")
        # Using absolute path or home dir is safer than './' in ROS nodes
        save_dir = "/home/sukhvansh/pose_recordings"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save as .npy
        np.save(os.path.join(save_dir, 'l_panda_poses.npy'), np.array(self.l_panda_poses))
        np.save(os.path.join(save_dir, 'r_panda_poses.npy'), np.array(self.r_panda_poses))
        np.save(os.path.join(save_dir, 'object_poses.npy'), np.array(self.object_poses))
        
        rospy.loginfo(f"Saved {len(self.object_poses)} frames to {save_dir}")


if __name__ == '__main__':
    try:
        recorder = PoseRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        pass