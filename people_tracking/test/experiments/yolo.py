#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class PeopleTracker:
    def __init__(self, name_subscriber_RGB: str) -> None:

        # Initialize YOLO
        model_path = "~/MEGA/developers/Donal/yolov8n-seg.pt"
        device = "cuda"
        self.model = YOLO(model_path).to(device)
        self.person_class = 0  # person class = 0

        # ROS Initialize
        rospy.init_node('listener', anonymous=True)
        self.publisher = rospy.Publisher('/hero/segmented_image', Image)
        self.subscriber = rospy.Subscriber(name_subscriber_RGB, Image, self.callback, queue_size = 1)
        self.cv_bridge = CvBridge()

    @staticmethod
    def detect(model, frame):
        """
        Return segemented image per class type.
        """
        results = model(frame)
        result = results[0]
        segmentation_contours_idx = [np.array(seg, dtype=np.int32) for seg in result.masks.xy]
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        return class_ids, segmentation_contours_idx

    def callback(self, data):
        rospy.loginfo("got message")
        seconds = rospy.get_time()
        cv_image = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        rospy.loginfo("converted message")

        classes, segmentations = self.detect(self.model, cv_image)

        mask = np.zeros_like(cv_image)

        for class_id, seg in zip(classes, segmentations):
            if class_id == self.person_class:
                # Fill the region enclosed by the polyline with white color (255)
                cv2.fillPoly(mask, [seg], (255, 255, 255))
        # Use the mask to cut out the regions from the original image
        cv_image[mask == 0] = 0  # Set the regions outside the mask to black (or any desired color)

        # # cv2.imshow("Segmented Image", cv_image)
        # # cv2.waitKey(1)

        image_message = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")

        self.publisher.publish(image_message)   # Send image with boundaries human


if __name__ == "__main__":
    laptop = True
    name_subscriber_RGB = "/hero/head_rgbd_sensor/rgb/image_raw" if not laptop else "video_frames"
    try:
        node_pt = PeopleTracker(name_subscriber_RGB)
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
