#!/usr/bin/env python
import rospy
import tf
import numpy as np
from sensor_msgs.msg import JointState
import os

class PoseRecorder:
    def __init__(self):
        rospy.init_node('pose_recorder', anonymous=True)

        self.l_panda_poses = []
        self.r_panda_poses = []
        self.object_poses = []

        self.listener = tf.TransformListener()

        self.joint_state_sub = rospy.Subscriber('/mujoco_ros/joint_states', JointState, self.joint_state_callback)

        rospy.on_shutdown(self.save_data)

        self.rate = rospy.Rate(2)

    def joint_state_callback(self, data):
        try:
            object_joint_index = data.name.index('object_to_grasp_joint')
            # The object pose is 7 values (x, y, z, qx, qy, qz, qw)
            pose_start_index = 0
            for i in range(object_joint_index):
                if data.name[i].endswith('joint1'): # Floating joints are one entry in the name list but have 7 position values
                    pose_start_index += 7
                else:
                    pose_start_index += 1

            object_pose = data.position[pose_start_index:pose_start_index+7]
            self.object_poses.append(object_pose)
        except ValueError:
            # Joint not found in this message
            pass

    def run(self):
        while not rospy.is_shutdown():
            try:
                # Get L_panda pose
                (trans_l, rot_l) = self.listener.lookupTransform('/world', '/L_panda_EE', rospy.Time(0))
                self.l_panda_poses.append(trans_l + rot_l)

                # Get R_panda pose
                (trans_r, rot_r) = self.listener.lookupTransform('/world', '/R_panda_EE', rospy.Time(0))
                self.r_panda_poses.append(trans_r + rot_r)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            self.rate.sleep()

    def save_data(self):
        rospy.loginfo("Saving recorded poses...")
        save_dir = "/home/sukhvansh/pose_recordings"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, 'l_panda_poses.npy'), np.array(self.l_panda_poses))
        np.save(os.path.join(save_dir, 'r_panda_poses.npy'), np.array(self.r_panda_poses))
        np.save(os.path.join(save_dir, 'object_poses.npy'), np.array(self.object_poses))
        rospy.loginfo("Saved poses to %s", save_dir)


if __name__ == '__main__':
    try:
        recorder = PoseRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        pass
