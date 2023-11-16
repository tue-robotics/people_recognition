#!/usr/bin/env python3
# Video from built-in webcam laptop
# Adapted from https://automaticaddison.com/working-with-ros-and-opencv-in-ros-noetic/

import sys
import rospy
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Header

laptop = sys.argv[1]

def publish_message():
    rospy.init_node('video_pub_py', anonymous=True)
    pub = rospy.Publisher('video_frames', Image, queue_size=10)

    rate = rospy.Rate(30)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)  # The argument '0' gets the default webcam.
    br = CvBridge()
    seq = 0
    while not rospy.is_shutdown():
        # Capture frame-by-frame. This method returns True/False as well as the video frame.
        ret, frame = cap.read()

        if ret:
            seq += 1
            header = Header()
            header.seq = seq
            header.stamp = rospy.Time.now()
            header.frame_id = 'camera_frame'  # You can modify the frame_id as needed

            # Create the Image message
            img_msg = Image()
            img_msg.header = header
            img_msg.height = frame.shape[0]
            img_msg.width = frame.shape[1]
            img_msg.encoding = 'bgr8'  # You may need to adjust the encoding based on your requirements
            img_msg.is_bigendian = 0
            img_msg.step = frame.shape[1] * 3  # Assuming 3 channels (BGR)

            # Flatten the image array and assign it to the data field
            img_msg.data = frame.flatten().tolist()

            pub.publish(img_msg)

        rate.sleep()

if __name__ == '__main__':
    if laptop == "True":
        try:
            publish_message()
        except rospy.ROSInterruptException:
            pass
        rospy.spin()
