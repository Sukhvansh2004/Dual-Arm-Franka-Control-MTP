#!/usr/bin/env python3

import rospy
import numpy as np
import tf.transformations as tft
import zmq
import threading

# --- ROS Imports ---
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError

# --- Import the service definition ---
from dual_arm_control.srv import GetGrasps, GetGraspsResponse

print("ROS, CV Bridge, and ZMQ imported successfully.")

class GraspGenServiceNode:
    """
    Provides a ROS service to get grasps for a specified arm.
    When called, it prompts a FastSAM node, waits for the resulting mask,
    and then calls the GraspGen ZMQ server to get grasps.
    """
    def __init__(self):
        rospy.init_node('graspgen_service_node')
        
        # --- Arm and Port configuration ---
        self.arm_id = rospy.get_param('~arm_id', 'L_panda')
        port = rospy.get_param('~port', 5557)
        rospy.loginfo(f"Starting GraspGen Service for arm: '{self.arm_id}' on port {port}.")

        # --- ZMQ Setup ---
        rospy.loginfo(f"Connecting to GraspGen server on tcp://localhost:{port}...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        self.zmq_lock = threading.Lock()
        rospy.loginfo(f"Connected to tcp://localhost:{port}")

        self.bridge = CvBridge()

        # --- ROS Params for GraspGen ---
        self.params = {
            'collision_threshold': rospy.get_param('~collision_threshold', 0.02),
            'max_scene_points': rospy.get_param('~max_scene_points', 8192),
            'num_grasps': rospy.get_param('~num_grasps', 200),
            'grasp_threshold': rospy.get_param('~grasp_threshold', 0.8),
            'top_k': rospy.get_param('~top_k', -1) # New parameter for top_k grasps
        }
        
        # --- Publishers ---
        # Publisher to send the prompt to the FastSAM node in the same namespace
        self.prompt_pub = rospy.Publisher('fastsam_client_node/fastsam/prompt', String, queue_size=1, latch=True)

        # --- Advertise the ROS Service ---
        service_name = f"get_grasps"
        rospy.Service(service_name, GetGrasps, self.handle_grasp_request)
        rospy.loginfo(f"Advertised grasp service at: {rospy.resolve_name(service_name)}")

        rospy.loginfo("GraspGen service node is ready.")

    def handle_grasp_request(self, req):
        rospy.loginfo(f"Grasp request received for arm '{self.arm_id}' with prompt: '{req.prompt}'")

        # --- 1. Publish the prompt to the FastSAM node ---
        self.prompt_pub.publish(String(data=req.prompt))
        rospy.loginfo(f"Prompt sent to '{self.prompt_pub.name}'. Waiting for segmentation...")
        # Add a small delay to ensure the fastsam_node has time to process and publish
        rospy.sleep(0.5) 

        # --- 2. Define topics and wait for messages ---
        base_topic = f"/mujoco_server/cameras/{self.arm_id}_camera_depth_frame"
        depth_topic = f"{base_topic}/depth/image_raw"
        info_topic = f"{base_topic}/depth/camera_info"
        # The mask topic is now relative to the node's namespace
        mask_topic = "fastsam_client_node/fastsam/mask" 

        try:
            rospy.loginfo(f"Waiting for sensor data and segmentation mask on topic {rospy.resolve_name(mask_topic)}...")
            depth_msg = rospy.wait_for_message(depth_topic, Image, timeout=5.0)
            cam_info = rospy.wait_for_message(info_topic, CameraInfo, timeout=5.0)
            mask_msg = rospy.wait_for_message(mask_topic, Image, timeout=10.0) # Longer timeout for segmentation
            rospy.loginfo("Received all required data.")
        except rospy.ROSException as e:
            error_msg = f"Failed to get required data: {e}"
            rospy.logerr(error_msg)
            return GetGraspsResponse(grasps=PoseArray(), success=False, message=error_msg)

        # --- 3. Process data ---
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            mask_image = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return GetGraspsResponse(grasps=PoseArray(), success=False, message=str(e))
        
        if np.sum(mask_image) == 0:
            rospy.logwarn("Received an empty mask from FastSAM. Aborting grasp prediction.")
            return GetGraspsResponse(grasps=PoseArray(), success=False, message="Segmentation failed (empty mask).")

        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32) / 1000.0

        # --- 4. Package data for ZMQ ---
        request_data = {
            'depth': depth_image,
            'mask': mask_image,
            'K': cam_info.K,
            'params': self.params
        }

        # --- 5. Send Request to GraspGen Server ---
        with self.zmq_lock:
            try:
                rospy.loginfo("Sending request to GraspGen server...")
                self.socket.send_pyobj(request_data)
                final_grasps = self.socket.recv_pyobj()
                rospy.loginfo(f"Received {len(final_grasps)} grasps from server.")
            except Exception as e:
                rospy.logerr(f"ZMQ request failed: {e}")
                return GetGraspsResponse(grasps=PoseArray(), success=False, message=str(e))

        if len(final_grasps) == 0:
            rospy.logwarn("No grasps returned from server.")
            return GetGraspsResponse(grasps=PoseArray(), success=False, message="No grasps found.")

        # --- 6. Convert to ROS Response ---
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
