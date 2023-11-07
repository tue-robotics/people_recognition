#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

# MSGS
from std_msgs.msg import String


NODE_NAME = 'HoC'
TOPIC_PREFIX = '/hero/'

class PeopleTracker:
    def __init__(self) -> None:

        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'HoC', String, queue_size= 2)
        # self.subscriber = rospy.Subscriber(name_subscriber_RGB, Image, self.callback, queue_size = 1)

        rate = rospy.Rate(2)
        # Keep publishing the messages until the user interrupts
        while not rospy.is_shutdown():
            message = "Hello World"
            # rospy.loginfo('Published: ' + message)
            # publish the message to the topic
            self.publisher.publish(message)
            rate.sleep()
    # def callback(self, data):
    #     self.publisher.publish("Hello World")
    #     rospy.loginfo("send message")


if __name__ == '__main__':
    try:
        node_pt = PeopleTracker()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
