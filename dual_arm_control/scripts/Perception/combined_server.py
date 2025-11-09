#!/usr/bin/env python

import numpy as np
import torch
import tf.transformations as tft
import cv2
import zmq # Import ZMQ
import sys
import traceback

# --- All your GraspGen/FastSAM imports ---
GRASPGEN_PATH = '/home/sukhvansh/franka_ws/src/GraspGen'
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
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed_h = points_h @ matrix.T
    return points_transformed_h[:, :3]
# --- END HELPER FUNCTION ---


class GraspPipelineServer:
    """
    Simplified server that runs FastSAM and GraspGen for a
    single, hardcoded finger gripper.
    """
    def __init__(self, finger_cfg_path):
        print("Loading FastSAM model...")
        self.sam_model = FastSAM('FastSAM-x.pt') # Using 'x' model as in your example
        print("FastSAM model loaded.")

        print(f"Loading HARDCODED GraspGen finger model from: {finger_cfg_path}")
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

    def run_prediction(self, color_image, depth_image, K, params, text_prompt=None):
        """
        The main pipeline, simplified for one gripper.
        """
        
        # --- 1. Get data ---
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        grasp_thresh = params.get('grasp_threshold', self.grasp_threshold)
        num_grasps = params.get('num_grasps', self.num_grasps)
        collision_thresh = params.get('collision_threshold', self.collision_threshold)

        # Create an empty mask template for failure cases
        empty_mask = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)
        empty_return = {'grasps': np.array([]), 'mask': empty_mask}

        # --- 2. Run FastSAM (UPDATED LOGIC) ---
        predict_args = {
            'source': color_image,
            'device': 'cuda',
            'retina_masks': True,
            'imgsz': 640,
            'conf': 0.4,
            'iou': 0.9,
            'verbose': False
        }

        if text_prompt: 
            print(f"[Server] Running FastSAM with text prompt: '{text_prompt}'")
            predict_args['texts'] = [text_prompt]
        else:
            print("[Server] Running FastSAM in 'everything' mode (no prompt).")
        
        results = self.sam_model.predict(**predict_args)
        
        if not results or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
            print("[Server] FastSAM did not detect any objects.")
            return empty_return

        masks_data = results[0].masks.data.cpu().numpy()
        final_mask_raw = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)

        if text_prompt:
            # Text prompt was given: Combine ALL returned masks
            print(f"[Server] Combining {len(masks_data)} masks from text prompt.")
            for mask_data in masks_data:
                resized_mask = cv2.resize(mask_data, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                binary_mask = (resized_mask > 0).astype(np.uint8)
                final_mask_raw = cv2.bitwise_or(final_mask_raw, binary_mask)
        
        else:
            # NO PROMPT: Find HIGHEST CONFIDENCE mask
            if results[0].probs is not None and len(results[0].probs) > 0:
                print(f"[Server] Finding highest confidence mask among {len(masks_data)} candidates.")
                confidences = results[0].probs.data.cpu().numpy()
                best_mask_index = np.argmax(confidences)
                best_mask_data = masks_data[best_mask_index]
                print(f"[Server] Highest confidence: {confidences[best_mask_index]:.2f}")
                resized_mask = cv2.resize(best_mask_data, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                final_mask_raw = (resized_mask > 0).astype(np.uint8)
            else:
                # Fallback to LARGEST AREA
                print(f"[Server] Confidence scores not available. Falling back to LARGEST AREA.")
                max_area = 0
                for mask_data in masks_data:
                    resized_mask = cv2.resize(mask_data, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (resized_mask > 0).astype(np.uint8)
                    area = np.sum(binary_mask)
                    if area > max_area:
                        max_area = area
                        final_mask_raw = binary_mask

        if np.sum(final_mask_raw) == 0:
            print("[Server] No valid masks found after processing.")
            return empty_return
        
        segmentation_mask_bool = final_mask_raw # This is 0/1
        segmentation_mask_img = (final_mask_raw * 255).astype(np.uint8) # This is 0/255
        empty_return = {'grasps': np.array([]), 'segmentation': results[0].plot()}

        # --- 3. Create point clouds ---
        try:
            scene_pc, object_pc, _, _ = depth_and_segmentation_to_point_clouds(
                depth_image=depth_image,
                segmentation_mask=segmentation_mask_bool, # Use the 0/1 mask
                fx=fx, fy=fy, cx=cx, cy=cy,
                target_object_id=1,
                remove_object_from_scene=True
            )
        except ValueError as e:
            print(f"[Server] Error creating point clouds: {e}")
            return empty_return
            
        if object_pc.shape[0] == 0:
            print("[Server] No object points found.")
            return empty_return

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
            return empty_return

        # --- 5. Run GraspGen Inference (Hardcoded for finger gripper) ---
        grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
            pc_filtered, self.gripper_sampler, grasp_thresh, num_grasps, topk_num_grasps=-1
        )

        if len(grasps_inferred) == 0:
            print("[Server] No grasps found from inference.")
            return empty_return
            
        grasps_inferred = grasps_inferred.cpu().numpy()

        # --- 6. Collision Filtering ---
        T_subtract_pc_mean = tft.translation_matrix(-pc_filtered.mean(axis=0))
        grasps_centered = np.array([T_subtract_pc_mean @ g for g in grasps_inferred])
        
        if scene_pc.shape[0] > 0:
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
        # final_grasps_centered = grasps_centered  # DISABLE COLLISION CHECK FOR TESTING

        if len(final_grasps_centered) == 0:
            print("[Server] All grasps filtered by collision.")
            return empty_return
            
        T_inv_subtract = tft.inverse_matrix(T_subtract_pc_mean)
        final_grasps = np.array([T_inv_subtract @ g for g in final_grasps_centered]) # Back to camera frame
        
        print(f"[Server] Found {len(final_grasps)} collision-free grasps.")
        
        # --- MODIFIED: Return dictionary ---
        return {'grasps': final_grasps, 'segmentation': results[0].plot()}

# --- Main execution (Simplified) ---
if __name__ == "__main__":
    # --- HARDCODED CONFIG ---
    FINGER_CONFIG = "/home/sukhvansh/franka_ws/src/GraspGenModels/checkpoints/graspgen_franka_panda.yml"
    
    pipeline = GraspPipelineServer(FINGER_CONFIG)
    
    # Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP) # REP = Reply
    socket.bind("tcp://*:5555")
    print("Grasp server started on tcp://*:5555")

    while True:
        try:
            print("Waiting for request...")
            request = socket.recv_pyobj()
            
            print(f"Processing request for {request['arm_id']}...")
            
            response_dict = pipeline.run_prediction(
                color_image=request['color'],
                depth_image=request['depth'],
                K=request['K'],
                params=request['params'],
                text_prompt=request.get('text_prompt', None) # Get prompt
            )
            
            socket.send_pyobj(response_dict)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            socket.send_pyobj({'grasps': np.array([]), 'mask': None})