#!/usr/bin/env python3
# Video from built-in webcam laptop
# Adapted from https://automaticaddison.com/working-with-ros-and-opencv-in-ros-noetic/

import sys
import rospy
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

laptop = sys.argv[1]


def publish_message():
    rospy.init_node('video_pub_py', anonymous=True)
    pub = rospy.Publisher('video_frames', Image, queue_size=10)

    rate = rospy.Rate(30)  # 10hz

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)  # The argument '0' gets the default webcam.
    br = CvBridge()

    while not rospy.is_shutdown():
        # Capture frame-by-frame. This method returns True/False as well as the video frame.
        ret, frame = cap.read()

        if ret:
            pub.publish(br.cv2_to_imgmsg(frame))

        rate.sleep()


if __name__ == '__main__':
    if laptop == "True":
        try:
            publish_message()
        except rospy.ROSInterruptException:
            pass
        rospy.spin()
