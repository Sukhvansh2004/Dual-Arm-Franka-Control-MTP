#!/usr/bin/env python3

import rospy
import numpy as np
import tf.transformations as tft
import tf
import zmq
import cv2
import sys

# --- ROS Imports ---
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError
print("ROS, CV Bridge, TF, and ZMQ imported successfully.")


class GraspPredictionNode:
    """
    Subscribes to {arm_id} topics, calls the server,
    and transforms the resulting grasps into the '{arm_id}_link0' frame.
    """
    def __init__(self):
        rospy.init_node('grasp_prediction_node_client', anonymous=True)
        rospy.loginfo("Starting Grasp Prediction Node (ZMQ Client).")

        # --- ZMQ Setup ---
        self.port = rospy.get_param('~port', 5555)

        rospy.loginfo(f"Connecting to GraspGen server on port {self.port}...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ) # REQ = Request
        socket_address = f"tcp://localhost:{self.port}"
        self.socket.connect(socket_address)
        rospy.loginfo(f"Connected to {socket_address}")
        
        # --- ADDED: TF Listener ---
        self.tf_listener = tf.TransformListener()

        # --- Get ROS Params ---
        self.params = {
            'collision_threshold': rospy.get_param('~collision_threshold', 0.02),
            'max_scene_points': rospy.get_param('~max_scene_points', 8192),
            'num_grasps': rospy.get_param('~num_grasps', 200),
            'grasp_threshold': rospy.get_param('~grasp_threshold', 0.8),
            'collision_check': rospy.get_param('~collision_check', True)
        }
        rospy.loginfo(f"Initialized with params {self.params}")

        # --- Hardcoded text prompt for testing ---
        self.text_prompt = "Wooden Block"
        if self.text_prompt:
             rospy.loginfo(f"Using text prompt: '{self.text_prompt}'")
        
        prediction_hz = rospy.get_param('~prediction_hz', 1)
        self.prediction_interval = rospy.Duration(1.0 / prediction_hz)

        try:
            self.arm_id = rospy.get_param('~arm_id')
        except KeyError:
            rospy.logerr("Fatal: Required parameter '~arm_id' not set.")
            rospy.logerr("Please specify the arm_id, e.g.: _arm_id:=L_panda")
            # We exit here because the node is useless without an arm_id
            sys.exit(1)

        # --- Dynamic: Define target TF frame ---
        self.target_frame = f"{self.arm_id}_link0"
        # We will get the camera frame from the cam_info message
        self.camera_frame = f"{self.arm_id}_camera_depth_frame" # Default
        
        rospy.loginfo(f"Configured for arm: {self.arm_id}. Target frame: {self.target_frame}")

        self.bridge = CvBridge()

        # --- Data Storage ---
        self.cam_info = None
        self.latest_depth = None
        self.latest_color = None
        self.last_prediction_time = rospy.Time(0)
        self.cam_info_sub = None # To store subscriber
        self.visualize = True

        # --- Publishers ---
        self.grasp_pub = rospy.Publisher(f"~{self.arm_id}/predicted_grasps", PoseArray, queue_size=1)
        
        if self.visualize:
            self.segmentation_pub = rospy.Publisher(f"~{self.arm_id}/segmentation", Image, queue_size=1)
        
        # --- Subscribers ---
        rospy.loginfo("Setting up subscribers...")
        base_topic = f"/mujoco_server/cameras/{self.arm_id}_camera_depth_frame"
        depth_topic = f"{base_topic}/depth/image_raw"
        color_topic = f"{base_topic}/rgb/image_raw"
        info_topic = f"{base_topic}/depth/camera_info"
        
        rospy.Subscriber(depth_topic, Image, self.depth_callback, queue_size=1)
        rospy.Subscriber(color_topic, Image, self.color_callback, queue_size=1)
        self.cam_info_sub = rospy.Subscriber(info_topic, CameraInfo, self.cam_info_callback)
            
        rospy.loginfo("Subscribers and publishers are set up. Waiting for data...")
        
        # --- Wait for TF ---
        try:
            rospy.loginfo(f"Waiting for transform between {self.target_frame} and {self.camera_frame}...")
            # Wait for the transform to be available once at the start
            self.tf_listener.waitForTransform(self.target_frame, self.camera_frame, rospy.Time(), rospy.Duration(10.0))
            rospy.loginfo("Transform is available!")
        except tf.Exception as e:
            rospy.logerr(f"Could not get transform in 10s. Is the robot_state_publisher running? Error: {e}")

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
            self.camera_frame = msg.header.frame_id # Get precise frame_id
            self.cam_info_sub.unregister()
            rospy.loginfo(f"Camera frame set to: {self.camera_frame}")

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
            'params': self.params,
            'text_prompt': self.text_prompt
        }

        # --- 3. Send Request and Wait for Reply ---
        try:
            rospy.loginfo(f"[{self.arm_id}] Sending request to server...")
            self.socket.send_pyobj(request_data)
            
            # --- MODIFIED: Receive dictionary ---
            response = self.socket.recv_pyobj()
            
            # Check for server error
            if response['segmentation'] is None:
                rospy.logerr(f"[{self.arm_id}] Server returned an error (segmentation is None).")
                return
            
            # Grasps are in the CAMERA frame
            grasps_in_camera_frame = response['grasps']
            
            rospy.loginfo(f"[{self.arm_id}] Received {len(grasps_in_camera_frame)} grasps from server.")
            
        except Exception as e:
            rospy.logerr(f"[{self.arm_id}] ZMQ request failed: {e}")
            return
            
        # --- 4. Get TF Transform ---
        try:
            # Get the transform from the camera frame to the hand frame
            (trans, rot) = self.tf_listener.lookupTransform(self.target_frame, self.camera_frame, rospy.Time(0))
            T_camera_to_hand = tft.concatenate_matrices(
                tft.translation_matrix(trans),
                tft.quaternion_matrix(rot)
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"[{self.arm_id}] TF error looking up transform from {self.camera_frame} to {self.target_frame}: {e}")
            return

        if len(grasps_in_camera_frame) == 0:
            rospy.logwarn(f"[{self.arm_id}] No grasps returned from server.")
            # Publish an empty array so RViz clears old poses
            empty_pose_array = PoseArray()
            empty_pose_array.header.stamp = rospy.Time.now()
            empty_pose_array.header.frame_id = self.target_frame
            self.grasp_pub.publish(empty_pose_array)
        else:
            # --- 7. Transform and Publish Grasps ---
            pose_array_msg = PoseArray()
            pose_array_msg.header.stamp = rospy.Time.now()
            pose_array_msg.header.frame_id = self.target_frame
            
            for P_camera in grasps_in_camera_frame:
                # --- APPLY TRANSFORM ---
                # P_hand = T_camera_to_hand @ P_camera
                P_hand = T_camera_to_hand @ P_camera
                
                p = Pose()
                trans = tft.translation_from_matrix(P_hand)
                quat = tft.quaternion_from_matrix(P_hand)
                p.position.x, p.position.y, p.position.z = trans
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
                pose_array_msg.poses.append(p)
                
            self.grasp_pub.publish(pose_array_msg)
            rospy.loginfo(f"[{self.arm_id}] Published {len(pose_array_msg.poses)} grasp poses IN {self.target_frame} FRAME.")
        
        # --- 8. 2D CV2 Visualization ---
        if self.visualize:
            self.segmentation_pub.publish(self.bridge.cv2_to_imgmsg(response['segmentation']))

if __name__ == '__main__':
    try:
        node = GraspPredictionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()