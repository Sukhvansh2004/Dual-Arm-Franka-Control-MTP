#!/home/sukhvansh/anaconda3/envs/GraspGen/bin/ python

import numpy as np
import torch # type: ignore
import tf.transformations as tft
import zmq
import sys
import traceback
import argparse

# --- All your GraspGen/FastSAM imports ---
GRASPGEN_PATH = '/home/sukhvansh/franka_ws/src/GraspGen' # Make sure this is correct
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
except ImportError as e:
    print(f"Error importing a library: {e}")
    print("Please run this in your 'GraspGen' conda environment.")
    sys.exit(1)

print("GraspGen libraries imported successfully.")

def transform_points(points, matrix):
    """
    Applies a 4x4 transformation matrix to a (N, 3) point cloud.
    """
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed_h = points_h @ matrix.T
    return points_transformed_h[:, :3]


class GraspPipelineServer:
    """
    Simplified server that runs ONLY GraspGen for a
    single, dynamically loaded finger gripper.
    """
    def __init__(self, finger_cfg_path):
        print(f"Loading GraspGen model from: {finger_cfg_path}")
        self.gripper_cfg = load_grasp_cfg(finger_cfg_path)
        self.gripper_sampler = GraspGenSampler(self.gripper_cfg)
        self.gripper_mesh = get_gripper_info(self.gripper_cfg.data.gripper_name).collision_mesh
        print("GraspGen models loaded.")

        # --- Default Parameters ---
        self.collision_threshold = 0.02
        self.max_scene_points = 8192
        self.num_grasps = 200
        self.grasp_threshold = 0.8
        self.max_object_points = 20000 # Safety cap for memory

    def run_prediction(self, depth_image, segmentation_mask, K, params):
        """
        The main pipeline, simplified for one gripper.
        """
        
        # --- 1. Get data ---
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        grasp_thresh = params.get('grasp_threshold', self.grasp_threshold)
        num_grasps = params.get('num_grasps', self.num_grasps)
        collision_thresh = params.get('collision_threshold', self.collision_threshold)
        top_k = params.get('top_k', -1) # Extract top_k parameter

        # --- 2. Create point clouds ---
        try:
            bool_mask = (segmentation_mask > 128).astype(np.uint8)
            
            scene_pc, object_pc, _, _ = depth_and_segmentation_to_point_clouds(
                depth_image=depth_image,
                segmentation_mask=bool_mask,
                fx=fx, fy=fy, cx=cx, cy=cy,
                target_object_id=1,
                remove_object_from_scene=True
            )
        except ValueError as e:
            print(f"[Server] Error creating point clouds: {e}")
            return np.array([])
            
        if object_pc.shape[0] == 0:
            print("[Server] No object points found from mask.")
            return np.array([])

        # --- 3. Filter object point cloud ---
        if object_pc.shape[0] > self.max_object_points:
            print(f"[Server] Object PC too large ({object_pc.shape[0]} points). Downsampling to {self.max_object_points}...")
            indices = np.random.choice(object_pc.shape[0], self.max_object_points, replace=False)
            object_pc = object_pc[indices]

        object_pc_torch = torch.from_numpy(object_pc)
        pc_filtered, _ = point_cloud_outlier_removal(object_pc_torch) 
        pc_filtered = pc_filtered.numpy()
        
        if pc_filtered.shape[0] < 50:
            print("[Server] Not enough points after filtering.")
            return np.array([])

        # --- 4. Run GraspGen Inference ---
        grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
            pc_filtered, self.gripper_sampler, grasp_thresh, num_grasps, topk_num_grasps=top_k
        )

        if len(grasps_inferred) == 0:
            print("[Server] No grasps found from inference.")
            return np.array([])
            
        grasps_inferred = grasps_inferred.cpu().numpy()

        # --- 5. Collision Filtering ---
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
            gripper_collision_mesh=self.gripper_mesh, 
            collision_threshold=collision_thresh,
        )
        
        final_grasps_centered = grasps_centered[collision_free_mask]
        
        if len(final_grasps_centered) == 0:
            print("[Server] All grasps filtered by collision.")
            return np.array([])
            
        T_inv_subtract = tft.inverse_matrix(T_subtract_pc_mean)
        final_grasps = np.array([T_inv_subtract @ g for g in final_grasps_centered]) 
        
        print(f"[Server] Found {len(final_grasps)} collision-free grasps.")
        return final_grasps

# --- Main execution (Now with arguments) ---
if __name__ == "__main__":
    # --- CONFIG PATHS ---
    # Assumes GraspGenModels is in the same src directory
    FINGER_CONFIG = "/home/sukhvansh/franka_ws/src/GraspGenModels/checkpoints/graspgen_franka_panda.yml"
    SUCTION_CONFIG = "/home/sukhvansh/franka_ws/src/GraspGenModels/checkpoints/graspgen_single_suction_cup_30mm.yml"

    parser = argparse.ArgumentParser(description="Run GraspGen ZMQ Server for a specific gripper type.")
    parser.add_argument(
        'gripper_type', 
        type=str, 
        choices=['finger', 'suction'],
        help="The type of gripper model to load ('finger' or 'suction')."
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=5557,
        help="The TCP port to bind the ZMQ server to (default: 5557)."
    )
    args = parser.parse_args()
    
    # --- Select config path based on arg ---
    if args.gripper_type == 'finger':
        config_path = FINGER_CONFIG
        print(f"Initializing server with FINGER gripper: {config_path}")
    elif args.gripper_type == 'suction':
        config_path = SUCTION_CONFIG
        print(f"Initializing server with SUCTION gripper: {config_path}")
    
    pipeline = GraspPipelineServer(config_path)
    
    # Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP) 
    socket_address = f"tcp://*:{args.port}"
    socket.bind(socket_address)
    print(f"GraspGen server started on {socket_address}")

    while True:
        try:
            print("GraspGen server waiting for request...")
            request = socket.recv_pyobj()
            
            print(f"Processing GraspGen request...")
            
            final_grasps = pipeline.run_prediction(
                depth_image=request['depth'],
                segmentation_mask=request['mask'],
                K=request['K'],
                params=request['params']
            )
            
            socket.send_pyobj(final_grasps)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            socket.send_pyobj(np.array([]))