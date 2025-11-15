#!/usr/bin/env python3

import rospy
import zmq
import numpy as np
import threading
import cv2 # type: ignore

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

class FastSAMClientNode:
    def __init__(self):
        rospy.init_node('fastsam_client_node')
        
        port = rospy.get_param('~port', 5556)
        
        # --- ZMQ Setup ---
        rospy.loginfo(f"[{rospy.get_name()}] Connecting to FastSAM server on tcp://localhost:{port}...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        self.zmq_lock = threading.Lock()
        rospy.loginfo(f"[{rospy.get_name()}] Connected to tcp://localhost:{port}")
        
        self.bridge = CvBridge()
        self.text_prompt = "a wooden hammer" # Default prompt
        self.prompt_lock = threading.Lock()

        # --- Get Topic Params ---
        default_topic = "/mujoco_server/cameras/L_panda_camera_depth_frame/rgb/image_raw"
        image_topic = rospy.get_param("~image_topic", default_topic)

        # --- Publishers ---
        self.mask_pub = rospy.Publisher("~fastsam/mask", Image, queue_size=1)
        self.viz_pub = rospy.Publisher("~fastsam/visualization", Image, queue_size=1)
        
        # --- Subscribers ---
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.prompt_sub = rospy.Subscriber("~fastsam/prompt", String, self.prompt_callback, queue_size=1)

        rospy.loginfo(f"[{rospy.get_name()}] Subscribed to image topic: {image_topic}")
        rospy.loginfo(f"[{rospy.get_name()}] Listening for prompts on: {rospy.resolve_name('~fastsam/prompt')}")
        rospy.loginfo(f"[{rospy.get_name()}] FastSAM client node running.")

    def prompt_callback(self, msg):
        with self.prompt_lock:
            self.text_prompt = msg.data
            rospy.loginfo(f"[{rospy.get_name()}] Received new prompt: '{self.text_prompt}'")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"[{rospy.get_name()}] CV Bridge error: {e}")
            return
            
        with self.prompt_lock:
            current_prompt = self.text_prompt
            
        request = {'image': cv_image, 'text_prompt': current_prompt}
        
        with self.zmq_lock:
            try:
                self.socket.send_pyobj(request)
                response = self.socket.recv_pyobj()
            except Exception as e:
                rospy.logerr(f"[{rospy.get_name()}] ZMQ request failed: {e}")    
                return
        
        if response.get('mask') is None:
            rospy.logwarn(f"[{rospy.get_name()}] FastSAM server returned an error or empty response.")
            return

        try:
            mask_msg = self.bridge.cv2_to_imgmsg(response['mask'], "mono8")
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)
            
            viz_image = response['viz']
            viz_msg = self.bridge.cv2_to_imgmsg(viz_image, "bgr8")
            viz_msg.header = msg.header
            self.viz_pub.publish(viz_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"[{rospy.get_name()}] CV Bridge error during republishing: {e}")
            
if __name__ == "__main__":
    try:
        node = FastSAMClientNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
