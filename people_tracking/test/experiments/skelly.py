#!/usr/bin/env python


import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


laptop = True
name_subscriber_RGB = '/hero/head_rgbd_sensor/rgb/image_raw' if not laptop else 'video_frames'


class PeopleTracker:
    def __init__(self) -> None:

        # Initialize YOLO
        model_path = "~/MEGA/developers/Donal/yolov8n-pose.pt"
        device = "cuda"
        self.model = YOLO(model_path).to(device)
        self.person_class = 0  # person class = 0

        # ROS Initialize
        rospy.init_node('listener', anonymous=True)
        self.publisher = rospy.Publisher('/hero/segmented_image', Image, queue_size = 1)
        self.subscriber = rospy.Subscriber(name_subscriber_RGB, Image, self.callback, queue_size = 1)

    @staticmethod
    def detect(model, frame):
        """
            Return segemented image per class type.
        """
        results = model(frame)
        result = results[0]

        return result

    def callback(self, data):
        rospy.loginfo("got message")
        seconds = rospy.get_time()
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        rospy.loginfo("converted message")

        result = self.detect(self.model, cv_image)

        plotted_results = result.plot()

        # # cv2.imshow("Segmented Image", cv_image)
        # # cv2.waitKey(1)

        image_message = bridge.cv2_to_imgmsg(plotted_results, encoding="passthrough")

        self.publisher.publish(image_message)   # Send image with boundaries human


if __name__ == '__main__':
    try:
        node_pt = PeopleTracker()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
