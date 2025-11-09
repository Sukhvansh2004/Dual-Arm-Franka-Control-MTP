#!/usr/bin/env python

import rospy
import numpy as np
import tf.transformations as tft
import zmq
import threading

# --- ROS Imports ---
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError

# --- Import the service definition ---
from dual_arm_control.srv import GetGrasps, GetGraspsResponse

print("ROS, CV Bridge, and ZMQ imported successfully.")

class GraspGenServiceNode:
    """
    Provides a ROS service to get grasps for L_panda.
    Subscribes to all required topics and calls the GraspGen ZMQ server
    when the service is requested.
    """
    def __init__(self):
        rospy.init_node('graspgen_service_node_test')
        rospy.loginfo("Starting GraspGen ROS Service Node (Test Mode).")

        # --- ZMQ Setup ---
        rospy.loginfo("Connecting to GraspGen server...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5557") # <-- NEW PORT
        self.zmq_lock = threading.Lock() # Lock is still good practice
        rospy.loginfo("Connected to tcp://localhost:5557")

        # --- Get ROS Params ---
        self.params = {
            'collision_threshold': 0.02,
            'max_scene_points': 8192,
            'num_grasps': 200,
            'grasp_threshold': 0.8
        }
        
        # --- HARDCODED for L_panda ---
        self.arm_id = "L_panda"
        rospy.loginfo(f"Hardcoded for arm: {self.arm_id}")

        self.bridge = CvBridge()

        # --- Data Storage (for latest messages) ---
        self.latest_depth = None
        self.latest_cam_info = None
        self.latest_mask = None
        self.cam_info_sub = None # To store subscriber for unregistering
        
        # --- Subscribers and Services ---
        rospy.loginfo("Setting up subscribers and services...")
        
        # --- Subscribe to required topics ---
        depth_topic = f"/mujoco_server/cameras/{self.arm_id}_camera_depth_frame/depth/image_raw"
        info_topic = f"/mujoco_server/cameras/{self.arm_id}_camera_depth_frame/depth/camera_info"
        # This topic comes from your fastsam_client_node
        mask_topic = f"/fastsam_client_{self.arm_id}/fastsam/mask" 

        rospy.Subscriber(depth_topic, Image, self.depth_callback, queue_size=1)
        rospy.Subscriber(mask_topic, Image, self.mask_callback, queue_size=1)
        self.cam_info_sub = rospy.Subscriber(info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
        
        # --- Advertise the ROS Service ---
        service_name = f"~{self.arm_id}/get_grasps"
        rospy.Service(service_name, GetGrasps, self.handle_grasp_request)
        rospy.loginfo(f"[{self.arm_id}] Advertised grasp service at: {service_name}")

        rospy.loginfo("GraspGen service node is ready.")

    # --- Subscriber Callbacks (just store data) ---
    def depth_callback(self, msg):
        self.latest_depth = msg

    def mask_callback(self, msg):
        self.latest_mask = msg

    def cam_info_callback(self, msg):
        if self.latest_cam_info is None:
            rospy.loginfo(f"[{self.arm_id}] Received camera info.")
            self.latest_cam_info = msg
            # Unsubscribe to save resources
            self.cam_info_sub.unregister()
            rospy.loginfo(f"[{self.arm_id}] Unsubscribed from camera_info.")

    # --- Service Handler (the main logic) ---
    def handle_grasp_request(self, req):
        rospy.loginfo(f"[{self.arm_id}] Grasp request received.")
        
        # --- 1. Check if all data is available ---
        if self.latest_depth is None:
            return GetGraspsResponse(grasps=PoseArray(), success=False, message=f"[{self.arm_id}] No depth data received yet.")
        if self.latest_cam_info is None:
            return GetGraspsResponse(grasps=PoseArray(), success=False, message=f"[{self.arm_id}] No camera info received yet.")
        if self.latest_mask is None:
            return GetGraspsResponse(grasps=PoseArray(), success=False, message=f"[{self.arm_id}] No mask data received from FastSAM node yet.")
            
        # --- 2. Copy data locally ---
        depth_msg = self.latest_depth
        mask_msg = self.latest_mask
        cam_info = self.latest_cam_info

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            mask_image = self.bridge.imgmsg_to_cv2(mask_msg, "mono8") # Mask is 8-bit
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return GetGraspsResponse(grasps=PoseArray(), success=False, message=str(e))
        
        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32) / 1000.0

        # --- 3. Package data for ZMQ ---
        request_data = {
            'depth': depth_image,
            'mask': mask_image,
            'K': cam_info.K,
            'params': self.params
        }

        # --- 4. Send Request and Wait for Reply (Protected by Lock) ---
        with self.zmq_lock:
            try:
                rospy.loginfo(f"[{self.arm_id}] Sending request to GraspGen server...")
                self.socket.send_pyobj(request_data)
                final_grasps = self.socket.recv_pyobj()
                rospy.loginfo(f"[{self.arm_id}] Received {len(final_grasps)} grasps from server.")
            except Exception as e:
                rospy.logerr(f"[{self.arm_id}] ZMQ request failed: {e}")
                return GetGraspsResponse(grasps=PoseArray(), success=False, message=str(e))

        if len(final_grasps) == 0:
            rospy.logwarn(f"[{self.arm_id}] No grasps returned from server.")
            return GetGraspsResponse(grasps=PoseArray(), success=False, message="No grasps found.")

        # --- 5. Convert to ROS Response ---
        pose_array_msg = PoseArray()
        pose_array_msg.header = depth_msg.header # Publish in the camera frame
        
        for grasp_matrix in final_grasps:
            p = Pose()
            trans = tft.translation_from_matrix(grasp_matrix)
            quat = tft.quaternion_from_matrix(grasp_matrix)
            p.position.x, p.position.y, p.position.z = trans
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            pose_array_msg.poses.append(p)
            
        return GetGraspsResponse(grasps=pose_array_msg, success=True, message="Success")

if __name__ == '__main__':
    try:
        node = GraspGenServiceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
