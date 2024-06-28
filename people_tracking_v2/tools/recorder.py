#!/usr/bin/env python

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
from datetime import datetime
from message_filters import Subscriber, ApproximateTimeSynchronizer

def save_image(cv_image, image_type, count, subfolder=None):
    # Create the base directory if it doesn't exist
    base_dir = 'ros/noetic/system/src/people_tracking_v2/data'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create the directory for today's date if it doesn't exist
    date_str = datetime.now().strftime('%a %b %d Test case 4')
    output_dir = os.path.join(base_dir, f'Frames {date_str}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories for RGB and Depth images
    if subfolder:
        image_dir = os.path.join(output_dir, subfolder)
    else:
        image_dir = os.path.join(output_dir, image_type)
        
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # Save the image
    filename = os.path.join(image_dir, f'{image_type}_{count:06d}.png')
    if image_type == 'depth' and subfolder is None:
        cv2.imwrite(filename, cv_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Ensure no compression to preserve depth values
    else:
        cv2.imwrite(filename, cv_image)
    rospy.loginfo(f'Saved {filename}')

def callback(rgb_msg, depth_msg):
    global rgb_count, depth_count
    try:
        # Convert ROS Image message to OpenCV image
        rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')  # Read as 16-bit

        # Save the RGB image
        save_image(rgb_image, 'rgb', rgb_count)
        
        # Save the depth image in original format and PNG format
        save_image(depth_image, 'depth', depth_count)
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_8bit = cv2.convertScaleAbs(depth_image_normalized)
        save_image(depth_image_8bit, 'depth', depth_count, subfolder='depth_png')
        
        rgb_count += 1
        depth_count += 1
    except CvBridgeError as e:
        rospy.logerr(f'CvBridge Error: {e}')

if __name__ == '__main__':
    rospy.init_node('image_saver', anonymous=True)

    # Initialize CvBridge
    bridge = CvBridge()

    # Initialize counters
    rgb_count = 0
    depth_count = 0

    # Define the topics
    rgb_topic = '/camera/color/image_raw'  # Topic for RGB images
    depth_topic = '/camera/depth/image_rect_raw'  # Topic for Depth images

    # Create subscribers
    rgb_sub = Subscriber(rgb_topic, Image)
    depth_sub = Subscriber(depth_topic, Image)

    # Synchronize the topics
    ats = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
    ats.registerCallback(callback)

    # Set the rate to 20 fps
    rate = rospy.Rate(20)

    try:
        while not rospy.is_shutdown():
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
