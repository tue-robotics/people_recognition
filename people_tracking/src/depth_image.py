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
        self.subscriber = rospy.Subscriber('/hero/head_rgbd_sensor/depth_registered/image_raw', Image, self.image_callback, queue_size=5)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'depth', Image, queue_size=5)
        self.depth_images = []
        self.depth_service = rospy.Service(TOPIC_PREFIX + NODE_NAME + '/depth_data', Depth, self.get_depth_data)

    def image_callback(self, data):
        """Store 5 seconds of depth data."""
        # self.depth_images = [img for img in self.depth_images if (rospy.Time.now() - img[0]).to_sec() <= 5]
        while self.depth_images and (float(rospy.get_time()) - self.depth_images[0][0]) > 5:
            self.depth_images.pop(0)
        # Store the current image
        self.depth_images.append([data.header.stamp.secs, data])

        msg = data
        self.publisher.publish(msg)

    def find_closest_index(self, desired_time):
        """Find the index of the closest image to the desired timestamp."""
        closest_index = None
        min_time_diff = float('inf')

        for i, (timestamp, _) in enumerate(self.depth_images):
            time_diff = abs((timestamp - desired_time))
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_index = i
        rospy.loginfo(f"closest_idx: {closest_index}, min_time_diff:{min_time_diff}, desired: {desired_time}")
        return closest_index

    def get_depth_data(self, data):
        """Get data from image and publish it to the topic."""
        rospy.loginfo(data)
        desired_time = data.desired_timestamp
        idx_image = self.find_closest_index(desired_time)

        if idx_image is not None:
            return self.depth_images[idx_image][1]
        else:
            rospy.logwarn("No depth image available.")
            return Image()

if __name__ == '__main__':
    try:
        node_di = DepthImage()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
