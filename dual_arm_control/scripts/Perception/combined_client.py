#!/usr/bin/env python

import rospy
import numpy as np
import tf.transformations as tft
import zmq
import cv2 # <-- Added for visualization

# --- ROS Imports ---
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError
print("ROS, CV Bridge, and ZMQ imported successfully.")


class GraspPredictionNode:
    """
    Subscribes to L_panda topics and calls the server.
    """
    def __init__(self):
        rospy.init_node('grasp_prediction_node_client')
        rospy.loginfo("Starting Grasp Prediction Node (ZMQ Client).")

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
        
        # --- NEW: Get text prompt from param ---
        self.text_prompt = "Wooden Handle"
        
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
            'params': self.params,
            'text_prompt': self.text_prompt # <-- ADDED PROMPT
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
            
            final_grasps = response['grasps']
            
            rospy.loginfo(f"[{self.arm_id}] Received {len(final_grasps)} grasps from server.")
            
        except Exception as e:
            rospy.logerr(f"[{self.arm_id}] ZMQ request failed: {e}")
            return
            
        # --- 4. Handle No Grasps ---
        if len(final_grasps) != 0:
            # --- 5. Publish Grasps ---
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
        else:
            rospy.logwarn(f"[{self.arm_id}] No grasps returned from server.")

        if self.visualize:
            # --- 6. 2D CV2 Visualization (Optional) ---
            # viz_image = color_image.copy()
            # K_matrix = np.array(cam_info.K).reshape(3, 3)
            # dist_coeffs = np.zeros(4) 
            # axis_points = np.float32([[0,0,0], [0.05,0,0], [0,0.05,0], [0,0,0.05]]).reshape(-1, 3)
            # for grasp_matrix in final_grasps:
            #     grasp_r_matrix = grasp_matrix[:3, :3]
            #     grasp_t_vector = grasp_matrix[:3, 3]
            #     grasp_r_vector, _ = cv2.Rodrigues(grasp_r_matrix)
            #     img_points, _ = cv2.projectPoints(axis_points, grasp_r_vector, grasp_t_vector, K_matrix, dist_coeffs)
                
            #     img_points = img_points.reshape(-1, 2).astype(int)
            #     p_origin = tuple(img_points[0])
            #     p_x = tuple(img_points[1])
            #     p_y = tuple(img_points[2])
            #     p_z = tuple(img_points[3])

            #     cv2.line(viz_image, p_origin, p_x, (0, 0, 255), 2) # X = Red
            #     cv2.line(viz_image, p_origin, p_y, (0, 255, 0), 2) # Y = Green
            #     cv2.line(viz_image, p_origin, p_z, (255, 0, 0), 2) # Z = Blue
            
            # Add the mask as an overlay
            self.segmentation_pub.publish(self.bridge.cv2_to_imgmsg(response['segmentation']))

if __name__ == '__main__':
    try:
        node = GraspPredictionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()