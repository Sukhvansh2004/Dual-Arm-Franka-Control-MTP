#!/usr/bin/env python

import rospy
import numpy as np
import tf.transformations as tft
import zmq

# --- ROS Imports ---
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError
print("ROS, CV Bridge, and ZMQ imported successfully.")


class GraspPredictionNode:
    """
    Simplified client for testing.
    Subscribes to L_panda topics and calls the server.
    """
    def __init__(self):
        rospy.init_node('grasp_prediction_node_client_test')
        rospy.loginfo("Starting Grasp Prediction Node (ZMQ Test Client).")

        # --- ZMQ Setup ---
        rospy.loginfo("Connecting to GraspGen server...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ) # REQ = Request
        self.socket.connect("tcp://localhost:5555")
        rospy.loginfo("Connected to tcp://localhost:5555")

        # --- Get ROS Params ---
        self.params = {
            'collision_threshold': rospy.get_param('~collision_threshold', 0.02),
            'max_scene_points': rospy.get_param('~max_scene_points', 8192),
            'num_grasps': rospy.get_param('~num_grasps', 200),
            'grasp_threshold': rospy.get_param('~grasp_threshold', 0.8)
        }
        
        prediction_hz = rospy.get_param('~prediction_hz', 0.5)
        self.prediction_interval = rospy.Duration(1.0 / prediction_hz)

        # --- HARDCODED for L_panda ---
        self.arm_id = "L_panda"
        rospy.loginfo(f"Hardcoded for arm: {self.arm_id}")

        self.bridge = CvBridge()

        # --- Data Storage ---
        self.cam_info = None
        self.latest_depth = None
        self.latest_color = None
        self.last_prediction_time = rospy.Time(0)
        self.cam_info_sub = None # To store subscriber

        # --- Publishers ---
        self.grasp_pub = rospy.Publisher(f"~{self.arm_id}/predicted_grasps", PoseArray, queue_size=1)
        
        # --- Subscribers ---
        rospy.loginfo("Setting up subscribers...")
        base_topic = f"/mujoco_server/cameras/{self.arm_id}_camera_depth_frame"
        depth_topic = f"{base_topic}/depth/image_raw"
        color_topic = f"{base_topic}/rgb/image_raw"
        info_topic = f"{base_topic}/depth/camera_info"
        
        rospy.Subscriber(depth_topic, Image, self.depth_callback)
        rospy.Subscriber(color_topic, Image, self.color_callback)
        self.cam_info_sub = rospy.Subscriber(info_topic, CameraInfo, self.cam_info_callback)
            
        rospy.loginfo("Subscribers and publishers are set up. Waiting for data...")

    def color_callback(self, msg):
        self.latest_color = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

        now = rospy.Time.now()
        if (now - self.last_prediction_time) < self.prediction_interval:
            return

        if self.latest_color is None or self.cam_info is None:
            rospy.logwarn_throttle(5.0, "Waiting for all data (color, depth, info)...")
            return
        
        self.last_prediction_time = now
        self.predict_grasps()

    def cam_info_callback(self, msg):
        if self.cam_info is None:
            rospy.loginfo(f"Received camera info for {self.arm_id} arm.")
            self.cam_info = msg
            # Unsubscribe to not waste resources
            self.cam_info_sub.unregister()

    def predict_grasps(self):
        rospy.loginfo(f"[{self.arm_id}] Preparing data for prediction server...")
        
        # --- 1. Get data and convert ---
        depth_msg = self.latest_depth
        color_msg = self.latest_color
        cam_info = self.cam_info

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return
        
        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32) / 1000.0

        # --- 2. Package data for ZMQ ---
        request_data = {
            'arm_id': self.arm_id,
            'color': color_image,
            'depth': depth_image,
            'K': cam_info.K,
            'params': self.params
        }

        # --- 3. Send Request and Wait for Reply ---
        try:
            rospy.loginfo(f"[{self.arm_id}] Sending request to server...")
            self.socket.send_pyobj(request_data)
            
            final_grasps = self.socket.recv_pyobj()
            rospy.loginfo(f"[{self.arm_id}] Received {len(final_grasps)} grasps from server.")
            
        except Exception as e:
            rospy.logerr(f"[{self.arm_id}] ZMQ request failed: {e}")
            return

        if len(final_grasps) == 0:
            rospy.logwarn(f"[{self.arm_id}] No grasps returned from server.")
            return

        # --- 4. Publish results ---
        pose_array_msg = PoseArray()
        pose_array_msg.header = depth_msg.header # Publish in the camera frame
        
        for grasp_matrix in final_grasps:
            p = Pose()
            trans = tft.translation_from_matrix(grasp_matrix)
            quat = tft.quaternion_from_matrix(grasp_matrix)
            p.position.x, p.position.y, p.position.z = trans
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            pose_array_msg.poses.append(p)
            
        self.grasp_pub.publish(pose_array_msg)
        rospy.loginfo(f"[{self.arm_id}] Published {len(pose_array_msg.poses)} grasp poses.")


if __name__ == '__main__':
    try:
        node = GraspPredictionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
