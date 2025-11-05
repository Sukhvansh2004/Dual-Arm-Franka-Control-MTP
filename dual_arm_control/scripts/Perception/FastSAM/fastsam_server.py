#!/home/sukhvansh/anaconda3/envs/GraspGen/bin/ python

import cv2
import zmq
import numpy as np
import sys
import traceback

try:
    from ultralytics import FastSAM
except ImportError as e:
    print(f"Error importing FastSAM: {e}")
    print("Please run this in your 'GraspGen' conda environment.")
    sys.exit(1)

class FastSAMServer:
    def __init__(self):
        print("Loading FastSAM model...")
        self.model = FastSAM('FastSAM-x.pt')
        print("FastSAM model loaded.")
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP) # REP = Reply
        self.socket.bind("tcp://*:5556") # Use a different port
        print("FastSAM server started on tcp://*:5556")

    def run(self):
        while True:
            try:
                # Wait for a request
                print("Waiting for image...")
                request = self.socket.recv_pyobj()
                
                color_image = request['image']
                text_prompt = request.get('text_prompt', None)

                # --- Build prediction arguments dynamically ---
                predict_args = {
                    'source': color_image,
                    'device': 'cuda',
                    'retina_masks': True,
                    'imgsz': 640,
                    'conf': 0.4, # This is the model's internal confidence threshold
                    'iou': 0.9,
                    'verbose': False,
                }

                if text_prompt: 
                    print(f"Running FastSAM with text prompt: '{text_prompt}'")
                    # Note: FastSAM's prompt argument is 'texts', not 'prompt'
                    predict_args['texts'] = [text_prompt]
                else:
                    print("Running FastSAM in 'everything' mode (no prompt).")
                
                results = self.model.predict(**predict_args)
                
                # --- Process Masks ---
                if not results or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
                    print("FastSAM did not detect any objects.")
                    self.socket.send_pyobj({
                        'mask': np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8),
                        'viz': color_image
                    })
                    continue

                masks_data = results[0].masks.data.cpu().numpy()
                final_mask_raw = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)

                if text_prompt:
                    # Text prompt was given: Combine ALL returned masks
                    print(f"Combining {len(masks_data)} masks from text prompt.")
                    for mask_data in masks_data:
                        resized_mask = cv2.resize(mask_data, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                        binary_mask = (resized_mask > 0).astype(np.uint8)
                        final_mask_raw = cv2.bitwise_or(final_mask_raw, binary_mask)
                
                else:
                    # Check if 'probs' data is available (it's not always)
                    if results[0].probs is not None and len(results[0].probs) > 0:
                        print(f"Finding highest confidence mask among {len(masks_data)} candidates.")
                        
                        # Get the index of the highest confidence score
                        confidences = results[0].probs.data.cpu().numpy()
                        best_mask_index = np.argmax(confidences)
                        best_mask_data = masks_data[best_mask_index]
                        
                        print(f"Highest confidence: {confidences[best_mask_index]:.2f}")

                        # Process the single best mask
                        resized_mask = cv2.resize(best_mask_data, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                        final_mask_raw = (resized_mask > 0).astype(np.uint8)
                    
                    else:
                        # Fallback to LARGEST AREA if confidence scores aren't available
                        print(f"Confidence scores not available. Falling back to LARGEST AREA.")
                        max_area = 0
                        for mask_data in masks_data:
                            resized_mask = cv2.resize(mask_data, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                            binary_mask = (resized_mask > 0).astype(np.uint8)
                            area = np.sum(binary_mask)
                            if area > max_area:
                                max_area = area
                                final_mask_raw = binary_mask

                mask_image = (final_mask_raw * 255).astype(np.uint8)
                
                # Get the annotated visualization image from FastSAM
                annotated_image = results[0].plot()

                # Send results back
                self.socket.send_pyobj({
                    'mask': mask_image,    # The 8-bit mask
                    'viz': annotated_image # The BGR image with overlays
                })

            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()
                self.socket.send_pyobj({'mask': None, 'viz': None})

if __name__ == "__main__":
    server = FastSAMServer()
    server.run()