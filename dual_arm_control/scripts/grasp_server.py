#!/usr/bin/env python

import numpy as np
import torch
import tf.transformations as tft
import cv2
import zmq # Import ZMQ
import sys
import traceback

# --- All your GraspGen/FastSAM imports ---
GRASPGEN_PATH = '/home/sukhvansh/GraspGen' # Make sure this is correct
if GRASPGEN_PATH not in sys.path:
    sys.path.append(GRASPGEN_PATH)

try:
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.utils.point_cloud_utils import (
        depth_and_segmentation_to_point_clouds, 
        point_cloud_outlier_removal, 
        filter_colliding_grasps
    )
    from grasp_gen.robot import get_gripper_info
    from ultralytics import FastSAM
except ImportError as e:
    print(f"Error importing a library: {e}")
    print("Please run this in your 'GraspGen' conda environment.")
    sys.exit(1)

print("All DL/GraspGen libraries imported successfully.")

# --- HELPER FUNCTION FOR TRANSFORMATIONS ---
def transform_points(points, matrix):
    """
    Applies a 4x4 transformation matrix to a (N, 3) point cloud.
    """
    # Convert (N, 3) points to (N, 4) homogeneous coordinates
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    # Apply transformation: (N, 4) @ (4, 4).T = (N, 4)
    points_transformed_h = points_h @ matrix.T
    # Convert back to (N, 3)
    return points_transformed_h[:, :3]
# --- END HELPER FUNCTION ---


class GraspPipelineServer:
    """
    Simplified server that runs FastSAM and GraspGen for a
    single, hardcoded finger gripper.
    """
    def __init__(self, finger_cfg_path):
        print("Loading FastSAM model...")
        self.sam_model = FastSAM('FastSAM-s.pt')
        print("FastSAM model loaded.")

        print(f"Loading HARECODED GraspGen finger model from: {finger_cfg_path}")
        self.gripper_cfg = load_grasp_cfg(finger_cfg_path)
        self.gripper_sampler = GraspGenSampler(self.gripper_cfg)
        self.gripper_mesh = get_gripper_info(self.gripper_cfg.data.gripper_name).collision_mesh
        print("GraspGen models loaded.")

        # --- Default Parameters (can be overridden by client) ---
        self.collision_threshold = 0.02
        self.max_scene_points = 8192
        self.num_grasps = 200
        self.grasp_threshold = 0.8
        self.max_object_points = 20000 # Safety cap for memory

    def run_prediction(self, color_image, depth_image, K, params):
        """
        The main pipeline, simplified for one gripper.
        """
        
        # --- 1. Get data ---
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        grasp_thresh = params.get('grasp_threshold', self.grasp_threshold)
        num_grasps = params.get('num_grasps', self.num_grasps)
        collision_thresh = params.get('collision_threshold', self.collision_threshold)

        # --- 2. Run FastSAM ---
        results = self.sam_model.predict(source=color_image, device='cuda', retina_masks=True, imgsz=640, conf=0.4, iou=0.9, verbose=False)
        
        if not results or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
            print("[Server] FastSAM did not detect any objects.")
            return np.array([]) 

        masks = results[0].masks.data.cpu().numpy()
        largest_mask = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)
        max_area = 0

        for mask_data in masks:
            resized_mask = cv2.resize(mask_data, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            binary_mask = (resized_mask > 0).astype(np.uint8)
            area = np.sum(binary_mask)
            if area > max_area:
                max_area = area
                largest_mask = binary_mask

        if max_area == 0:
            print("[Server] No valid masks found.")
            return np.array([])
        
        segmentation_mask = largest_mask

        # --- 3. Create point clouds ---
        try:
            scene_pc, object_pc, _, _ = depth_and_segmentation_to_point_clouds(
                depth_image=depth_image,
                segmentation_mask=segmentation_mask,
                fx=fx, fy=fy, cx=cx, cy=cy,
                target_object_id=1,
                remove_object_from_scene=True
            )
        except ValueError as e:
            print(f"[Server] Error creating point clouds: {e}")
            return np.array([])
            
        if object_pc.shape[0] == 0:
            print("[Server] No object points found.")
            return np.array([])

        # --- 4. Filter object point cloud (MEMORY FIX) ---
        if object_pc.shape[0] > self.max_object_points:
            print(f"[Server] Object PC too large ({object_pc.shape[0]} points). Downsampling to {self.max_object_points}...")
            indices = np.random.choice(object_pc.shape[0], self.max_object_points, replace=False)
            object_pc = object_pc[indices]

        object_pc_torch = torch.from_numpy(object_pc)
        pc_filtered, _ = point_cloud_outlier_removal(object_pc_torch) # This is now safe
        pc_filtered = pc_filtered.numpy()
        
        if pc_filtered.shape[0] < 50:
            print("[Server] Not enough points after filtering.")
            return np.array([])

        # --- 5. Run GraspGen Inference (Hardcoded for finger gripper) ---
        grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
            pc_filtered, self.gripper_sampler, grasp_thresh, num_grasps, topk_num_grasps=-1
        )

        if len(grasps_inferred) == 0:
            print("[Server] No grasps found from inference.")
            return np.array([])
            
        grasps_inferred = grasps_inferred.cpu().numpy()

        # --- 6. Collision Filtering (ATTRIBUTE_ERROR FIX) ---
        T_subtract_pc_mean = tft.translation_matrix(-pc_filtered.mean(axis=0))
        grasps_centered = np.array([T_subtract_pc_mean @ g for g in grasps_inferred])
        
        if scene_pc.shape[0] > 0:
            # Use our helper function, not tft.transform_points
            scene_pc_centered = transform_points(scene_pc, T_subtract_pc_mean) 
            if len(scene_pc_centered) > self.max_scene_points:
                indices = np.random.choice(len(scene_pc_centered), self.max_scene_points, replace=False)
                scene_pc_downsampled = scene_pc_centered[indices]
            else:
                scene_pc_downsampled = scene_pc_centered
        else:
            scene_pc_downsampled = np.array([]).reshape(0, 3) 

        collision_free_mask = filter_colliding_grasps(
            scene_pc=scene_pc_downsampled,
            grasp_poses=grasps_centered,
            gripper_collision_mesh=self.gripper_mesh, # Hardcoded mesh
            collision_threshold=collision_thresh,
        )
        
        final_grasps_centered = grasps_centered[collision_free_mask]
        
        if len(final_grasps_centered) == 0:
            print("[Server] All grasps filtered by collision.")
            return np.array([])
            
        T_inv_subtract = tft.inverse_matrix(T_subtract_pc_mean)
        final_grasps = np.array([T_inv_subtract @ g for g in final_grasps_centered]) # Back to camera frame
        
        print(f"[Server] Found {len(final_grasps)} collision-free grasps.")
        return final_grasps

# --- Main execution (Simplified) ---
if __name__ == "__main__":
    # --- HARDCODED CONFIG ---
    FINGER_CONFIG = "/home/sukhvansh/GraspGenModels/checkpoints/graspgen_franka_panda.yml"
    
    pipeline = GraspPipelineServer(FINGER_CONFIG)
    
    # Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP) # REP = Reply
    socket.bind("tcp://*:5555")
    print("Grasp server (Test Mode) started on tcp://*:5555")

    while True:
        try:
            print("Waiting for request...")
            request = socket.recv_pyobj()
            
            print(f"Processing request for {request['arm_id']}...")
            
            final_grasps = pipeline.run_prediction(
                color_image=request['color'],
                depth_image=request['depth'],
                K=request['K'],
                params=request['params']
            )
            
            socket.send_pyobj(final_grasps)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            socket.send_pyobj(np.array([]))
