#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

from UKFclass import *

# MSGS
from people_tracking.msg import ColourCheckedTarget
from sensor_msgs.msg import Image


NODE_NAME = 'people_tracker'
TOPIC_PREFIX = '/hero/'

laptop = True
name_subscriber_RGB = '/hero/head_rgbd_sensor/rgb/image_raw' if not laptop else 'video_frames'


class PeopleTracker:
    def __init__(self) -> None:

        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber = rospy.Subscriber(TOPIC_PREFIX + 'HoC', ColourCheckedTarget, self.callback, queue_size=1)
        # self.publisher = rospy.Publisher(TOPIC_PREFIX + 'Location', String, queue_size= 2)

        self.subscriber_debug = rospy.Subscriber(name_subscriber_RGB, Image, self.plot_tracker, queue_size=1)
        self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'people_tracker_debug', Image, queue_size=10)

        # Variables
        self.ukf = UKF()

    def plot_tracker(self, data):
        latest_image = data
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(latest_image, desired_encoding='passthrough')

        self.ukf.predict(float(rospy.get_time()))
        x_position = int(self.ukf.kf.x[0])
        # rospy.loginfo('predict: time:  ' + str(float(rospy.get_time())) + 'x: ' + str(x_position))

        x_position = 0 if x_position < 0 else x_position
        x_position = 639 if x_position > 639 else x_position
        cv2.circle(cv_image, (x_position, 200), 5, (0, 0, 255), -1)
        tracker_image = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        self.publisher_debug.publish(tracker_image)



    def callback(self, data):
        x_position = data.x_position
        time = data.time
        position = [x_position, 0]
        rospy.loginfo('time: ' + str(time) + ' x: ' +str(x_position))
        self.ukf.update(time, position)




if __name__ == '__main__':
    try:
        node_pt = PeopleTracker()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
