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
from dual_arm_control.srv import GetGrasps, GetGraspsResponse

print("ROS, CV Bridge, TF, and ZMQ imported successfully.")


class GraspPredictionService:
    """
    Provides a ROS service to trigger grasp prediction.
    On service call, it gathers sensor data, calls the GraspGen ZMQ server,
    and publishes the resulting grasps to the appropriate topics.
    """
    def __init__(self):
        rospy.init_node('grasp_prediction_service_node', anonymous=True)
        rospy.loginfo("Starting Grasp Prediction Service Node.")

        # --- ZMQ Setup ---
        self.port = rospy.get_param('~port', 5555)
        rospy.loginfo(f"Connecting to GraspGen server on port {self.port}...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        socket_address = f"tcp://localhost:{self.port}"
        self.socket.connect(socket_address)
        rospy.loginfo(f"Connected to {socket_address}")
        
        self.tf_listener = tf.TransformListener()
        self.bridge = CvBridge()

        # --- Get ROS Params for grasp prediction ---
        self.params = {
            'collision_threshold': rospy.get_param('~collision_threshold', 0.02),
            'max_scene_points': rospy.get_param('~max_scene_points', 8192),
            'num_grasps': rospy.get_param('~num_grasps', 200),
            'grasp_threshold': rospy.get_param('~grasp_threshold', 0.8),
            'collision_check': rospy.get_param('~collision_check', True),
            'top_k': rospy.get_param('~top_k', 1)
        }
        rospy.loginfo(f"Initialized with params {self.params}")

        self.visualize = True

        # --- Publishers (created on demand and stored) ---
        self.grasp_pubs = {}
        self.segmentation_pubs = {}

        # --- Service ---
        self.service = rospy.Service('predict_grasps', GetGrasps, self.handle_predict_grasps)
        rospy.loginfo("Service 'predict_grasps' is ready.")

    def get_publisher(self, topic, msg_type, pubs_dict):
        """Creates a publisher if it doesn't exist, otherwise returns the existing one."""
        if topic not in pubs_dict:
            rospy.loginfo(f"Creating publisher for topic: {topic}")
            pubs_dict[topic] = rospy.Publisher(topic, msg_type, queue_size=1, latch=True)
        return pubs_dict[topic]

    def handle_predict_grasps(self, req):
        """
        Service handler to perform grasp prediction for a given arm.
        """
        arm_id = req.arm_id
        text_prompt = req.text_prompt
        rospy.loginfo(f"Received grasp prediction request for arm: '{arm_id}' with prompt: '{text_prompt}'")

        # --- Define topics based on arm_id ---
        base_topic = f"/mujoco_server/cameras/{arm_id}_camera_depth_frame"
        depth_topic = f"{base_topic}/depth/image_raw"
        color_topic = f"{base_topic}/rgb/image_raw"
        info_topic = f"{base_topic}/depth/camera_info"

        # --- Wait for one-time messages ---
        try:
            rospy.loginfo("Waiting for sensor data...")
            depth_msg = rospy.wait_for_message(depth_topic, Image, timeout=5.0)
            color_msg = rospy.wait_for_message(color_topic, Image, timeout=5.0)
            cam_info = rospy.wait_for_message(info_topic, CameraInfo, timeout=5.0)
            rospy.loginfo("Received all sensor data.")
        except rospy.ROSException as e:
            error_msg = f"Failed to get sensor data: {e}"
            rospy.logerr(error_msg)
            return GetGraspsResponse(success=False, message=error_msg)

        # --- TF Setup ---
        target_frame = f"{arm_id}_link0"
        camera_frame = cam_info.header.frame_id

        try:
            rospy.loginfo(f"Waiting for transform between '{target_frame}' and '{camera_frame}'...")
            self.tf_listener.waitForTransform(target_frame, camera_frame, rospy.Time(), rospy.Duration(5.0))
            rospy.loginfo("Transform is available.")
        except tf.Exception as e:
            error_msg = f"Could not get transform: {e}. Is robot_state_publisher running?"
            rospy.logerr(error_msg)
            return GetGraspsResponse(success=False, message=error_msg)

        # --- Process and Predict ---
        rospy.loginfo(f"[{arm_id}] Preparing data for prediction server...")
        
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        except CvBridgeError as e:
            error_msg = f"CV Bridge error: {e}"
            rospy.logerr(error_msg)
            return GetGraspsResponse(success=False, message=error_msg)
        
        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32) / 1000.0

        request_data = {
            'arm_id': arm_id,
            'color': color_image,
            'depth': depth_image,
            'K': cam_info.K,
            'params': self.params,
            'text_prompt': text_prompt
        }

        try:
            rospy.loginfo(f"[{arm_id}] Sending request to ZMQ server...")
            self.socket.send_pyobj(request_data)
            response = self.socket.recv_pyobj()
            
            # Robust check for server error
            if response.get('grasps') is None or (response.get('segmentation') is None and response.get('mask') is None):
                error_msg = f"[{arm_id}] Server returned an error or incomplete data."
                rospy.logerr(error_msg)
                return GetGraspsResponse(success=False, message=error_msg)
            
            grasps_in_camera_frame = response['grasps']
            rospy.loginfo(f"[{arm_id}] Received {len(grasps_in_camera_frame)} grasps from server.")
            
        except Exception as e:
            error_msg = f"[{arm_id}] ZMQ request failed: {e}"
            rospy.logerr(error_msg)
            # Attempt to reconnect for the next call
            self.socket.close()
            self.socket = zmq.Context().socket(zmq.REQ)
            self.socket.connect(f"tcp://localhost:{self.port}")
            return GetGraspsResponse(success=False, message=error_msg)
            
        try:
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, camera_frame, rospy.Time(0))
            T_camera_to_target = tft.concatenate_matrices(
                tft.translation_matrix(trans),
                tft.quaternion_matrix(rot)
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            error_msg = f"[{arm_id}] TF error looking up transform: {e}"
            rospy.logerr(error_msg)
            return GetGraspsResponse(success=False, message=error_msg)

        # --- Publish Grasps ---
        grasp_pub_topic = f"/{arm_id}/cartesian_impedance_example_controller/equilibrium_pose"
        grasp_pub = self.get_publisher(grasp_pub_topic, PoseArray, self.grasp_pubs)
        
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = target_frame

        if len(grasps_in_camera_frame) == 0:
            rospy.logwarn(f"[{arm_id}] No grasps returned from server. Publishing empty array.")
        else:
            # Transform all valid grasps from camera to target frame
            for grasp_matrix_camera in grasps_in_camera_frame:
                grasp_matrix_target = T_camera_to_target @ grasp_matrix_camera
                
                p = Pose()
                trans = tft.translation_from_matrix(grasp_matrix_target)
                quat = tft.quaternion_from_matrix(grasp_matrix_target)
                p.position.x, p.position.y, p.position.z = trans
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
                pose_array_msg.poses.append(p)
        
        grasp_pub.publish(pose_array_msg)
        rospy.loginfo(f"[{arm_id}] Published {len(pose_array_msg.poses)} grasp poses to '{grasp_pub_topic}'.")
        
        # --- Publish Visualization ---
        if self.visualize and response.get('segmentation') is not None:
            seg_pub_topic = f"~{arm_id}/segmentation"
            seg_pub = self.get_publisher(seg_pub_topic, Image, self.segmentation_pubs)
            try:
                seg_img_msg = self.bridge.cv2_to_imgmsg(response['segmentation'], "bgr8")
                seg_pub.publish(seg_img_msg)
            except CvBridgeError as e:
                rospy.logwarn(f"Could not publish segmentation image: {e}")


        success_msg = f"Successfully processed request for {arm_id}. Published {len(pose_array_msg.poses)} grasps."
        return GetGraspsResponse(success=True, message=success_msg)

if __name__ == '__main__':
    try:
        GraspPredictionService()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
