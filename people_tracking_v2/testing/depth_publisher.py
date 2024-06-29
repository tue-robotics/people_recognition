#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import numpy as np

def publish_depth_images_from_folder(folder_path, topic_name):
    pub = rospy.Publisher(topic_name, Image, queue_size=10)
    bridge = CvBridge()

    # List all files in the folder and sort them
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    total_frames = len(image_files)

    rate = rospy.Rate(18.7)  # Set the desired rate

    for image_file in image_files:
        if rospy.is_shutdown():
            break

        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read as is (e.g., 16-bit depth)

        if frame is None:
            rospy.logerr(f"Error reading image file {image_path}")
            continue

        try:
            if frame.dtype == np.uint16:
                # Resize the image to 1280x720
                frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_NEAREST)
                image_message = bridge.cv2_to_imgmsg(frame_resized, "16UC1")
            else:
                rospy.logwarn(f"Unexpected image format for depth image: {image_file}")
                continue
            pub.publish(image_message)
            rate.sleep()
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('depth_publisher_node', anonymous=True)
    folder_path = '/home/miguel/Documents/BEP-Testing/Test Case 1/Frames Sat Jun 29 Test Case 1/depth' #/home/miguel/Documents/BEP-Testing/Test Case 1/Frames Sat Jun 29 Test Case 1/depth
    topic_name = '/hero/head_rgbd_sensor/depth_registered/image_raw'
    try:
        publish_depth_images_from_folder(folder_path, topic_name)
    except rospy.ROSInterruptException:
        pass
