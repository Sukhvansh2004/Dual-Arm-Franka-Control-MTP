#!/usr/bin/env python3

import rospy
import numpy as np
import tf.transformations as tft
import tf
import zmq
import threading
import sys

# --- ROS Imports ---
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from cv_bridge import CvBridge, CvBridgeError

# --- Import the service definition ---
from dual_arm_control.srv import GetGrasps, GetGraspsResponse

print("ROS, CV Bridge, TF, and ZMQ imported successfully.")

class GraspGenServiceNode:
    """
    Provides a ROS service to get grasps for a specified arm.
    When called, it prompts a FastSAM node, waits for the resulting mask,
    calls the GraspGen ZMQ server, transforms the grasps, and publishes them.
    """
    def __init__(self):
        rospy.init_node('graspgen_service_node')
        
        # --- Arm and Port configuration ---
        self.arm_id = rospy.get_param('~arm_id', 'L_panda')
        port = rospy.get_param('~port', 5557)
        rospy.loginfo(f"[{rospy.get_name()}] Starting GraspGen Service for arm: '{self.arm_id}' on port {port}.")

        # --- ZMQ Setup ---
        rospy.loginfo(f"[{rospy.get_name()}] Connecting to GraspGen server on tcp://localhost:{port}...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        self.zmq_lock = threading.Lock()
        rospy.loginfo(f"[{rospy.get_name()}] Connected to tcp://localhost:{port}")

        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()

        # --- Gripper Correction Setup ---
        gripper_type = rospy.get_param('~gripper_type', 'finger')
        if gripper_type == 'finger':
            rospy.loginfo(f"[{rospy.get_name()}] Using FINGER gripper correction.")
            _corr_trans = [0.0, 0.0, 0.0]
            _corr_quat = [0.0, 0.0, -0.7071, 0.7071]
        elif gripper_type == 'suction':
            rospy.loginfo(f"[{rospy.get_name()}] Using SUCTION gripper correction.")
            _corr_trans = [0.0, 0.0, 0.11]
            _corr_quat = [1.0, 0.0, 0.0, 0.0]
        else:
            rospy.logfatal(f"[{rospy.get_name()}] Unknown gripper_type '{gripper_type}'")
            sys.exit(1)

        self.T_grasp_correction = tft.concatenate_matrices(
            tft.translation_matrix(_corr_trans),
            tft.quaternion_matrix(_corr_quat)
        )

        # --- ROS Params for GraspGen ---
        self.params = {
            'collision_threshold': rospy.get_param('~collision_threshold', 0.02),
            'max_scene_points': rospy.get_param('~max_scene_points', 8192),
            'num_grasps': rospy.get_param('~num_grasps', 200),
            'grasp_threshold': rospy.get_param('~grasp_threshold', 0.8),
            'top_k': rospy.get_param('~top_k', -1)
        }
        
        # --- Publishers ---
        self.prompt_pub = rospy.Publisher('fastsam_client_node/fastsam/prompt', String, queue_size=1, latch=True)
        self.inferred_poses_pub = rospy.Publisher("inferred_grasp_poses", PoseArray, queue_size=1, latch=True)
        self.single_grasp_pub = rospy.Publisher("cartesian_impedance_example_controller/equilibrium_pose", PoseStamped, queue_size=1, latch=True)

        # --- Advertise the ROS Service ---
        service_name = f"~get_grasps"
        rospy.Service(service_name, GetGrasps, self.handle_grasp_request)
        rospy.loginfo(f"[{rospy.get_name()}] Advertised grasp service at: {rospy.resolve_name(service_name)}")

        rospy.loginfo(f"[{rospy.get_name()}] GraspGen service node is ready.")

    def handle_grasp_request(self, req):
        rospy.loginfo(f"[{rospy.get_name()}] Grasp request received for arm '{self.arm_id}' with prompt: '{req.prompt}'")

        # --- 1. Publish the prompt to the FastSAM node ---
        self.prompt_pub.publish(String(data=req.prompt))
        rospy.loginfo(f"[{rospy.get_name()}] Prompt sent to '{self.prompt_pub.name}'. Waiting for segmentation...")
        # Add a small delay to ensure the fastsam_node has time to process and publish
        rospy.sleep(0.5) 

        # --- 2. Define topics and wait for messages ---
        base_topic = f"/mujoco_server/cameras/{self.arm_id}_camera_depth_frame"
        depth_topic = f"{base_topic}/depth/image_raw"
        info_topic = f"{base_topic}/depth/camera_info"
        # The mask topic is now relative to the node's namespace
        mask_topic = "fastsam_client_node/fastsam/mask" 

        try:
            rospy.loginfo(f"[{rospy.get_name()}] Waiting for sensor data and segmentation mask on topic {rospy.resolve_name(mask_topic)}...")
            depth_msg = rospy.wait_for_message(depth_topic, Image, timeout=5.0)
            cam_info = rospy.wait_for_message(info_topic, CameraInfo, timeout=5.0)
            mask_msg = rospy.wait_for_message(mask_topic, Image, timeout=10.0) # Longer timeout for segmentation
            rospy.loginfo(f"[{rospy.get_name()}] Received all required data.")
        except rospy.ROSException as e:
            error_msg = f"[{rospy.get_name()}] Failed to get required data: {e}"
            rospy.logerr(error_msg)
            return GetGraspsResponse(success=False, message=error_msg)

        # --- 3. Process data ---
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            mask_image = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")
        except CvBridgeError as e:
            rospy.logerr(f"[{rospy.get_name()}] CV Bridge error: {e}")
            return GetGraspsResponse(success=False, message=str(e))
        
        if np.sum(mask_image) == 0:
            rospy.logwarn(f"[{rospy.get_name()}] Received an empty mask from FastSAM. Aborting grasp prediction.")
            return GetGraspsResponse(success=False, message="Segmentation failed (empty mask).")

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
                rospy.loginfo(f"[{rospy.get_name()}] Sending request to GraspGen server...")
                self.socket.send_pyobj(request_data)
                final_grasps = self.socket.recv_pyobj()
                rospy.loginfo(f"[{rospy.get_name()}] Received {len(final_grasps)} grasps from server.")
            except Exception as e:
                rospy.logerr(f"[{rospy.get_name()}] ZMQ request failed: {e}")
                return GetGraspsResponse(success=False, message=str(e))

        if len(final_grasps) == 0:
            rospy.logwarn(f"[{rospy.get_name()}] No grasps returned from server.")
            # Still publish an empty array so downstream nodes know there are no grasps
            empty_pose_array = PoseArray()
            empty_pose_array.header.stamp = rospy.Time.now()
            empty_pose_array.header.frame_id = f"{self.arm_id}_link0"
            self.inferred_poses_pub.publish(empty_pose_array)
            return GetGraspsResponse(success=False, message="No grasps found.")

        # --- 6. Transform Grasps to Robot Frame ---
        target_frame = f"{self.arm_id}_link0"
        camera_frame = cam_info.header.frame_id

        try:
            rospy.loginfo(f"[{rospy.get_name()}] Waiting for transform between '{target_frame}' and '{camera_frame}'...")
            self.tf_listener.waitForTransform(target_frame, camera_frame, rospy.Time(), rospy.Duration(5.0))
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, camera_frame, rospy.Time(0))
            T_target_camera = tft.concatenate_matrices(
                tft.translation_matrix(trans),
                tft.quaternion_matrix(rot)
            )
            rospy.loginfo(f"[{rospy.get_name()}] Transform is available.")
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            error_msg = f"[{rospy.get_name()}] TF error looking up transform: {e}"
            rospy.logerr(error_msg)
            return GetGraspsResponse(success=False, message=error_msg)

        # --- 7. Convert to ROS Response and Publish ---
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = target_frame
        
        for grasp_matrix_camera in final_grasps:
            # Apply transforms: base_link <- camera_link <- grasp <- correction
            grasp_matrix_in_target = T_target_camera @ grasp_matrix_camera @ self.T_grasp_correction
            
            p = Pose()
            trans = tft.translation_from_matrix(grasp_matrix_in_target)
            quat = tft.quaternion_from_matrix(grasp_matrix_in_target)
            p.position.x, p.position.y, p.position.z = trans
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            pose_array_msg.poses.append(p)
        
        # Publish the full array of transformed grasps
        self.inferred_poses_pub.publish(pose_array_msg)
        rospy.loginfo(f"[{rospy.get_name()}] Published {len(pose_array_msg.poses)} grasp poses to '{self.inferred_poses_pub.name}'.")

        # Publish the first valid grasp (if any) to the controller topic
        if pose_array_msg.poses:
            first_grasp_pose = PoseStamped(header=pose_array_msg.header, pose=pose_array_msg.poses[0])
            self.single_grasp_pub.publish(first_grasp_pose)
            rospy.loginfo(f"[{rospy.get_name()}] Published the first grasp pose to '{self.single_grasp_pub.name}'.")

        return GetGraspsResponse(success=True, message="Success")

if __name__ == '__main__':
    try:
        node = GraspGenServiceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
