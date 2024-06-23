#!/usr/bin/env python

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
from datetime import datetime

def save_image(cv_image, image_type, count):
    # Create the base directory if it doesn't exist
    base_dir = 'ros/noetic/system/src/people_tracking_v2/data'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create the directory for today's date if it doesn't exist
    date_str = datetime.now().strftime('%a %b %d 1')
    output_dir = os.path.join(base_dir, f'Frames {date_str}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories for RGB and Depth images
    image_dir = os.path.join(output_dir, image_type)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # Save the image
    filename = os.path.join(image_dir, f'{image_type}_{count:06d}.png')
    cv2.imwrite(filename, cv_image)
    rospy.loginfo(f'Saved {filename}')

def rgb_callback(msg):
    global rgb_count
    # Convert ROS Image message to OpenCV image
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    # Save the image
    save_image(cv_image, 'rgb', rgb_count)
    rgb_count += 1

def depth_callback(msg):
    global depth_count
    # Convert ROS Image message to OpenCV image
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # Normalize and convert to 8-bit
    depth_image_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_8bit = cv2.convertScaleAbs(depth_image_normalized)
    # Save the image
    save_image(depth_image_8bit, 'depth', depth_count)
    depth_count += 1

if __name__ == '__main__':
    rospy.init_node('image_saver', anonymous=True)

    # Initialize CvBridge
    bridge = CvBridge()

    # Initialize counters
    rgb_count = 0
    depth_count = 0

    # Define the topics
    rgb_topic = '/hero/head_rgbd_sensor/rgb/image_raw'  # Topic for RGB images
    depth_topic = '/hero/head_rgbd_sensor/depth_registered/image_raw' # Topic for Depth images

    # Subscribe to the topics
    rospy.Subscriber(rgb_topic, Image, rgb_callback)
    rospy.Subscriber(depth_topic, Image, depth_callback)

    # Keep the node running
    rospy.spin()
