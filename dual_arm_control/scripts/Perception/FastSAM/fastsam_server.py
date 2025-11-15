#!/usr/bin/env python3

import cv2 # type: ignore
import zmq
import numpy as np
import sys
import traceback
import argparse

try:
    from ultralytics import FastSAM # type: ignore
except ImportError as e:
    print(f"Error importing FastSAM: {e}")
    print("Please run this in your 'GraspGen' conda environment.")
    sys.exit(1)

class FastSAMServer:
    def __init__(self):
        print("Loading FastSAM model...")
        self.model = FastSAM('FastSAM-x.pt')
        print("FastSAM model loaded.")
        
    def run(self, port):
        # ZMQ setup
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket_address = f"tcp://*:{port}"
        socket.bind(socket_address)
        print(f"FastSAM server started on {socket_address}")

        while True:
            try:
                # Wait for a request
                print("Waiting for image and prompt...")
                request = socket.recv_pyobj()
                
                color_image = request['image']
                text_prompt = request.get('text_prompt', None)

                # --- Build prediction arguments ---
                predict_args = {
                    'source': color_image,
                    'device': 'cuda',
                    'retina_masks': True,
                    'imgsz': 640,
                    'conf': 0.8,
                    'iou': 0.9,
                    'verbose': False,
                }

                if text_prompt: 
                    print(f"Running FastSAM with text prompt: '{text_prompt}'")
                    predict_args['texts'] = [text_prompt]
                else:
                    print("Running FastSAM in 'everything' mode (no prompt).")
                
                results = self.model.predict(**predict_args)
                
                # --- Process Masks ---
                if not results or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
                    print("FastSAM did not detect any objects.")
                    socket.send_pyobj({
                        'mask': np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8),
                        'viz': color_image
                    })
                    continue

                masks_data = results[0].masks.data.cpu().numpy()
                final_mask_raw = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)

                # Always combine masks if a text prompt is given
                for mask_data in masks_data:
                    resized_mask = cv2.resize(mask_data, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (resized_mask > 0).astype(np.uint8)
                    final_mask_raw = cv2.bitwise_or(final_mask_raw, binary_mask)

                mask_image = (final_mask_raw * 255).astype(np.uint8)
                annotated_image = results[0].plot()

                # Send results back
                socket.send_pyobj({
                    'mask': mask_image,
                    'viz': annotated_image
                })

            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()
                # Ensure a reply is always sent
                socket.send_pyobj({'mask': None, 'viz': None})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastSAM ZMQ server.")
    parser.add_argument(
        '--port', 
        type=int, 
        default=5556,
        help="The TCP port to bind the ZMQ server to (default: 5556)."
    )
    args = parser.parse_args()

    server = FastSAMServer()
    server.run(args.port)
