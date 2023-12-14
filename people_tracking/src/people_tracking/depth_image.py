#!/usr/bin/env python
import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

# MSGS
from sensor_msgs.msg import Image
from people_tracking.srv import Depth

NODE_NAME = 'depth'
TOPIC_PREFIX = '/hero/'


class DepthImage:
    def __init__(self) -> None:
        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber = rospy.Subscriber('/hero/head_rgbd_sensor/depth_registered/image_raw', Image, self.image_callback, queue_size=2)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'depth', Image, queue_size=2)
        self.depth_images = []
        self.depth_service = rospy.Service(TOPIC_PREFIX + NODE_NAME + '/depth_data', Depth, self.get_depth_data)
        self.bridge = CvBridge()

    def image_callback(self, data, time_data_stored_sec: int = 60):
        """Store recent depth data for given amount of time."""
        if data is None:
            rospy.logwarn("Received NoneType data in image_callback.")
            return

        # while self.depth_images and (float(rospy.get_time()) - self.depth_images[0][0]) > time_data_stored_sec:
        #     self.depth_images.pop(0)

        # Store the current image
        # cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.depth_images.append([data.header.stamp.secs, data])
        rospy.loginfo("depth")

        # msg = data
        self.publisher.publish(data)

    def find_closest_index(self, desired_time):
        """Find the index of the closest image to the desired timestamp."""
        closest_image = None
        min_time_diff = float('inf')

        for i, (timestamp, image) in enumerate(self.depth_images):
            time_diff = abs((timestamp - desired_time))
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_image = image
        # rospy.loginfo(f"closest_idx: , min_time_diff:{min_time_diff}, desired: {desired_time}")
        return closest_image

    def get_depth_data(self, data):
        """Get data from image and publish it to the topic if data available."""
        # desired_time = data.desired_timestamp
        # desired_image = self.find_closest_index(desired_time)
        #
        # if desired_image is not None:
        #     bridge = CvBridge()
        #     depth_image_msg = bridge.cv2_to_imgmsg(desired_image, encoding="passthrough")
        #     return depth_image_msg
        # else:
        #     rospy.logwarn("No depth image available.")
        #     return Image()

        return self.bridge.cv2_to_imgmsg(self.depth_images[-1][1], encoding="passthrough")


if __name__ == '__main__':
    try:
        node_di = DepthImage()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
