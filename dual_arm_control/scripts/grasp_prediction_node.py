#!/usr/bin/env python

import rospy
import numpy as np
import torch
import tf.transformations as tft
import cv2

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError
print("ROS and CV Bridge imported successfully.")

# It is recommended to run this node from a sourced terminal where the GraspGen
# conda environment is activated. If not, you may need to add the path manually.
import sys
GRASPGEN_PATH = '/home/sukhvansh/GraspGen' # This path is from your user context
if GRASPGEN_PATH not in sys.path:
    sys.path.append(GRASPGEN_PATH)

try:
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg # pyright: ignore[reportMissingImports]
    from grasp_gen.utils.point_cloud_utils import depth_and_segmentation_to_point_clouds, point_cloud_outlier_removal, filter_colliding_grasps # pyright: ignore[reportMissingImports]
    from grasp_gen.robot import get_gripper_info # pyright: ignore[reportMissingImports]
    print("GraspGen library imported successfully.")
except ImportError as e:
    print(f"Error importing GraspGen: {e}")
    print("Please ensure GRASPGEN_PATH is correct and you have activated the correct conda environment.")
    sys.exit(1)

# FastSAM Integration
try:
    from ultralytics import FastSAM
    print("FastSAM imported successfully.")
except ImportError as e:
    print(f"Error importing FastSAM: {e}")
    sys.exit(1)


class GraspPredictionNode:
    """
    A ROS node to perform grasp prediction using the GraspGen library.
    It listens for depth and color images from arm-mounted cameras,
    runs FastSAM for segmentation, runs GraspGen inference, and
    publishes the predicted grasp poses for each arm.
    """
    def __init__(self):
        rospy.init_node('grasp_prediction_node')
        rospy.loginfo("Starting Grasp Prediction Node.")

        # --- Parameters ---
        self.suction_config_path = rospy.get_param('~suction_gripper_config')
        self.gripper_config_path = rospy.get_param('~finger_gripper_config')
        self.collision_threshold = rospy.get_param('~collision_threshold', 0.02)
        self.max_scene_points = rospy.get_param('~max_scene_points', 8192)
        self.num_grasps = rospy.get_param('~num_grasps', 200)
        self.grasp_threshold = rospy.get_param('~grasp_threshold', 0.8)
        
        # --- Pipeline Control Parameters ---
        prediction_hz = rospy.get_param('~prediction_hz', 0.5) # Default: 1 inference every 2 seconds
        self.prediction_interval = rospy.Duration(1.0 / prediction_hz)
        self.depth_ignore_threshold = rospy.get_param('~depth_ignore_threshold', 0.02) # 2cm

        # --- NEW: Arm and Gripper Configuration ---
        self.arm_id_0 = rospy.get_param('~arm_id_0', 'L_panda')
        self.arm_id_1 = rospy.get_param('~arm_id_1', 'R_panda')
        self.arm_ids = [self.arm_id_0, self.arm_id_1]
        
        use_suction_0 = rospy.get_param('~use_suction_gripper_0', False)
        use_suction_1 = rospy.get_param('~use_suction_gripper_1', True)
        
        # Map arm_id to gripper type ('suction' or 'finger')
        self.arm_gripper_map = {
            self.arm_id_0: 'suction' if use_suction_0 else 'finger',
            self.arm_id_1: 'suction' if use_suction_1 else 'finger'
        }
        rospy.loginfo(f"Arm configuration: {self.arm_gripper_map}")

        # --- CV Bridge ---
        self.bridge = CvBridge()

        # --- FastSAM Model ---
        rospy.loginfo("Loading FastSAM model...")
        self.sam_model = FastSAM('FastSAM-s.pt')
        rospy.loginfo("FastSAM model loaded.")

        # --- GraspGen Samplers ---
        rospy.loginfo("Loading GraspGen models...")
        self.suction_cfg = load_grasp_cfg(self.suction_config_path)
        self.gripper_cfg = load_grasp_cfg(self.gripper_config_path)
        self.suction_sampler = GraspGenSampler(self.suction_cfg)
        self.gripper_sampler = GraspGenSampler(self.gripper_cfg)
        
        self.suction_gripper_mesh = get_gripper_info(self.suction_cfg.data.gripper_name).collision_mesh
        self.finger_gripper_mesh = get_gripper_info(self.gripper_cfg.data.gripper_name).collision_mesh
        rospy.loginfo("GraspGen models loaded.")

        # --- Data Storage (Keyed by arm_id) ---
        self.cam_info = {}
        self.latest_depth = {}
        self.latest_color = {}
        self.last_prediction_time = {}

        # --- Publishers (Keyed by arm_id) ---
        self.grasp_pubs = {}
        
        # --- Subscribers ---
        rospy.loginfo("Setting up subscribers and publishers...")
        for arm_id in self.arm_ids:
            # Initialize data storage for this arm
            self.cam_info[arm_id] = None
            self.latest_depth[arm_id] = None
            self.latest_color[arm_id] = None
            self.last_prediction_time[arm_id] = rospy.Time(0)

            # Construct topic names based on your provided list
            base_topic = f"/mujoco_server/cameras/{arm_id}_camera_depth_frame"
            depth_topic = f"{base_topic}/depth/image_raw"
            color_topic = f"{base_topic}/rgb/image_raw"
            info_topic = f"{base_topic}/depth/camera_info" # Use DEPTH camera info for depth image
            
            # Create publishers
            pub_topic = f"~{arm_id}/predicted_grasps"
            self.grasp_pubs[arm_id] = rospy.Publisher(pub_topic, PoseArray, queue_size=1)
            rospy.loginfo(f"  [{arm_id}] Publishing grasps to: {pub_topic}")

            # Create subscribers, passing arm_id as callback_args
            rospy.Subscriber(depth_topic, Image, self.depth_callback, callback_args=arm_id)
            rospy.Subscriber(color_topic, Image, self.color_callback, callback_args=arm_id)
            rospy.Subscriber(info_topic, CameraInfo, self.cam_info_callback, callback_args=arm_id)
            
            rospy.loginfo(f"  [{arm_id}] Subscribing to:")
            rospy.loginfo(f"    Depth: {depth_topic}")
            rospy.loginfo(f"    Color: {color_topic}")
            rospy.loginfo(f"    Info:  {info_topic}")

        rospy.loginfo("Subscribers and publishers are set up. Waiting for data...")

    def color_callback(self, msg, arm_id):
        self.latest_color[arm_id] = msg

    def depth_callback(self, msg, arm_id):
        """Main callback to trigger grasp prediction, with rate limiting."""
        self.latest_depth[arm_id] = msg

        # Rate limit the prediction
        now = rospy.Time.now()
        last_time = self.last_prediction_time[arm_id]
        if (now - last_time) < self.prediction_interval:
            return

        # Check if all required data is available
        if self.latest_color[arm_id] is None:
            rospy.logwarn(f"[{arm_id}] Triggered prediction but no color image yet.")
            return
        if self.cam_info[arm_id] is None:
            rospy.logwarn(f"[{arm_id}] Triggered prediction but no camera info yet.")
            return
        
        self.last_prediction_time[arm_id] = now
        self.predict_grasps(arm_id)

    def cam_info_callback(self, msg, arm_id):
        if self.cam_info[arm_id] is None:
            rospy.loginfo(f"Received camera info for {arm_id} arm.")
        self.cam_info[arm_id] = msg

    def predict_grasps(self, arm_id):
        rospy.loginfo(f"[{arm_id}] Predicting grasps...")
        
        # --- 1. Get data and convert ---
        depth_msg = self.latest_depth[arm_id]
        color_msg = self.latest_color[arm_id]
        cam_info = self.cam_info[arm_id]

        try:
            # Ensure depth is read as 32-bit float (meters) or 16-bit int (mm)
            # MuJoCo depth is usually 32FC1 (meters)
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return
        
        # Handle potential depth encoding issues (e.g., if it's 16UC1 in mm)
        if depth_image.dtype == np.uint16:
            rospy.logwarn_once(f"[{arm_id}] Depth image is 16UC1, assuming millimeters and converting to meters.")
            depth_image = depth_image.astype(np.float32) / 1000.0
        elif depth_image.dtype != np.float32:
            rospy.logwarn(f"[{arm_id}] Unknown depth image type: {depth_image.dtype}. Trying to convert to float32.")
            depth_image = depth_image.astype(np.float32)


        # --- 2. Pre-process depth image ---
        # Ignore points that are too close (likely the gripper)
        # depth_image[depth_image < self.depth_ignore_threshold] = 0 # This might be too aggressive

        # --- 3. Run FastSAM to get segmentation mask ---
        rospy.loginfo(f"[{arm_id}] Running FastSAM segmentation...")
        try:
            results = self.sam_model.predict(source=color_image, device='cuda', retina_masks=True, imgsz=640, conf=0.4, iou=0.9, verbose=False)
        except Exception as e:
            rospy.logerr(f"[{arm_id}] FastSAM prediction failed: {e}")
            return
            
        if not results or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
            rospy.logwarn(f"[{arm_id}] FastSAM did not detect any objects.")
            return

        # Find the largest mask and assume it's the target object
        masks = results[0].masks.data.cpu().numpy()
        largest_mask = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)
        max_area = 0

        for mask_data in masks:
            # Resize mask to match image dimensions
            resized_mask = cv2.resize(mask_data, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            binary_mask = (resized_mask > 0).astype(np.uint8)
            area = np.sum(binary_mask)
            if area > max_area:
                max_area = area
                largest_mask = binary_mask # The object ID is implicitly 1

        if max_area == 0:
            rospy.logwarn(f"[{arm_id}] No valid masks found after processing FastSAM results.")
            return
        
        segmentation_mask = largest_mask
        rospy.loginfo(f"[{arm_id}] Found largest mask with area: {max_area}")

        # --- 4. Get camera intrinsics ---
        K = cam_info.K
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

        # --- 5. Create point clouds ---
        try:
            scene_pc, object_pc, _, _ = depth_and_segmentation_to_point_clouds(
                depth_image=depth_image,
                segmentation_mask=segmentation_mask,
                fx=fx, fy=fy, cx=cx, cy=cy,
                target_object_id=1, # The largest mask is our target object with ID 1
                remove_object_from_scene=True
            )
        except ValueError as e:
            rospy.logerr(f"[{arm_id}] Error creating point clouds: {e}")
            return
            
        if object_pc.shape[0] == 0:
            rospy.logwarn(f"[{arm_id}] No object points found after segmentation.")
            return
        if scene_pc.shape[0] == 0:
            rospy.logwarn(f"[{arm_id}] No scene points found. This may cause issues.")

        # --- 6. Filter object point cloud ---
        object_pc_torch = torch.from_numpy(object_pc)
        pc_filtered, _ = point_cloud_outlier_removal(object_pc_torch)
        pc_filtered = pc_filtered.numpy()
        
        if pc_filtered.shape[0] < 50: # Need a minimum number of points
            rospy.logwarn(f"[{arm_id}] Not enough points in object PC after filtering: {pc_filtered.shape[0]}")
            return

        # --- 7. Run GraspGen Inference ---
        gripper_type = self.arm_gripper_map[arm_id]
        sampler = self.suction_sampler if gripper_type == 'suction' else self.gripper_sampler
        
        grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
            pc_filtered,
            sampler,
            grasp_threshold=self.grasp_threshold,
            num_grasps=self.num_grasps,
            topk_num_grasps=-1 # No topk filtering here
        )

        if len(grasps_inferred) == 0:
            rospy.logwarn(f"[{arm_id}] No grasps found from inference.")
            return
            
        grasps_inferred = grasps_inferred.cpu().numpy()

        # --- 8. Collision Filtering ---
        rospy.loginfo(f"[{arm_id}] Filtering {len(grasps_inferred)} grasps for collisions.")
        
        T_subtract_pc_mean = tft.translation_matrix(-pc_filtered.mean(axis=0))
        grasps_centered = np.array([T_subtract_pc_mean @ g for g in grasps_inferred])
        
        # Only use scene_pc if it's not empty
        if scene_pc.shape[0] > 0:
            scene_pc_centered = tft.transform_points(scene_pc, T_subtract_pc_mean)
            if len(scene_pc_centered) > self.max_scene_points:
                indices = np.random.choice(len(scene_pc_centered), self.max_scene_points, replace=False)
                scene_pc_downsampled = scene_pc_centered[indices]
            else:
                scene_pc_downsampled = scene_pc_centered
        else:
            scene_pc_downsampled = np.array([]).reshape(0, 3) # Empty array

        gripper_mesh = self.suction_gripper_mesh if gripper_type == 'suction' else self.finger_gripper_mesh
        
        collision_free_mask = filter_colliding_grasps(
            scene_pc=scene_pc_downsampled,
            grasp_poses=grasps_centered,
            gripper_collision_mesh=gripper_mesh,
            collision_threshold=self.collision_threshold,
        )
        
        final_grasps_centered = grasps_centered[collision_free_mask]
        
        if len(final_grasps_centered) == 0:
            rospy.logwarn(f"[{arm_id}] All grasps were filtered out due to collisions.")
            return
            
        T_inv_subtract = tft.inverse_matrix(T_subtract_pc_mean)
        final_grasps = np.array([T_inv_subtract @ g for g in final_grasps_centered])

        rospy.loginfo(f"[{arm_id}] Found {len(final_grasps)} collision-free grasps.")

        # --- 9. Publish results ---
        pose_array_msg = PoseArray()
        pose_array_msg.header = depth_msg.header # Publish in the camera frame
        
        for grasp_matrix in final_grasps:
            p = Pose()
            trans = tft.translation_from_matrix(grasp_matrix)
            quat = tft.quaternion_from_matrix(grasp_matrix)
            p.position.x, p.position.y, p.position.z = trans
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            pose_array_msg.poses.append(p)
            
        self.grasp_pubs[arm_id].publish(pose_array_msg)
        rospy.loginfo(f"[{arm_id}] Published {len(pose_array_msg.poses)} grasp poses.")


if __name__ == '__main__':
    try:
        node = GraspPredictionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"An unhandled exception occurred: {e}")
        import traceback
        traceback.print_exc()