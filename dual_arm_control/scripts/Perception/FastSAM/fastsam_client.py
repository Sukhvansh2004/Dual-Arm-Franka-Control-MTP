#!/usr/bin/env python3

import rospy
import zmq
import numpy as np
import threading
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class FastSAMClientNode:
    def __init__(self):
        rospy.init_node('fastsam_client_node')
        
        # --- ZMQ Setup ---
        rospy.loginfo("Connecting to FastSAM server...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5556")
        self.zmq_lock = threading.Lock()
        rospy.loginfo("Connected to tcp://localhost:5556")
        
        self.bridge = CvBridge()

        # --- Get Topic Params ---
        default_topic = "/mujoco_server/cameras/L_panda_camera_depth_frame/rgb/image_raw"
        image_topic = rospy.get_param("~image_topic", default_topic)
        
        # --- NEW: VISUALIZATION PARAM ---
        self.visualize = rospy.get_param("~visualize_cv2", True)
        if self.visualize:
            rospy.loginfo("CV2 visualization is ON.")

        # --- Publishers ---
        self.mask_pub = rospy.Publisher("~fastsam/mask", Image, queue_size=1)
        self.viz_pub = rospy.Publisher("~fastsam/visualization", Image, queue_size=1)
        
        # --- Subscriber ---
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo(f"Subscribed to {image_topic}")
        rospy.loginfo("FastSAM client node running.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return
            
        # --- NEW: SHOW LIVE FEED ---
        if self.visualize:
            cv2.imshow("Live Feed (Client)", cv_image)
            cv2.waitKey(1) # This is crucial for updating the window
            
        request = {'image': cv_image, 'text_prompt': 'Wooden Hammer with metal head'}
        
        with self.zmq_lock:
            try:
                self.socket.send_pyobj(request)
                response = self.socket.recv_pyobj()
            except Exception as e:
                rospy.logerr(f"ZMQ request failed: {e}")    
                return
        
        if response['mask'] is None:
            rospy.logwarn("FastSAM server returned an error.")
            return

        try:
            mask_msg = self.bridge.cv2_to_imgmsg(response['mask'], "mono8")
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)
            
            viz_image = response['viz'] # This is the BGR image
            viz_msg = self.bridge.cv2_to_imgmsg(viz_image, "bgr8")
            viz_msg.header = msg.header
            self.viz_pub.publish(viz_msg)
            
            if self.visualize:
                cv2.imshow("FastSAM Result (Client)", viz_image)
                cv2.waitKey(1) # Also crucial
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error during republishing: {e}")
            
if __name__ == "__main__":
    try:
        node = FastSAMClientNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        # --- NEW: Clean up windows ---
        cv2.destroyAllWindows()