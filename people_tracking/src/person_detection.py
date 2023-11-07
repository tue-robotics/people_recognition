#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from cv_bridge import CvBridge

# MSGS
from sensor_msgs.msg import Image
from people_tracking.msg import DetectedPerson

NODE_NAME = 'person_detection'
TOPIC_PREFIX = '/hero/'

laptop = True
name_subscriber_RGB = '/hero/head_rgbd_sensor/rgb/image_raw' if not laptop else 'video_frames'


class PeopleTracker:
    def __init__(self) -> None:
        # Initialize YOLO
        model_path = "~/MEGA/developers/Donal/yolov8n-seg.pt"
        device = "cuda"
        self.model = YOLO(model_path).to(device)
        self.person_class = 0  # person class = 0

        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber = rospy.Subscriber(name_subscriber_RGB, Image, self.image_callback, queue_size=1)
        # self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'segmented_image', Image, queue_size=10)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'person_detections', DetectedPerson, queue_size= 10)

        # Initialize variables
        self.batch_nr = 0
        self.latest_image = None  # To store the most recent image
        self.latest_image_time = None


    def image_callback(self, data):
        """ Make sure that only the most recent data will be processed."""
        # Cancel any previously queued image processing tasks
        if self.latest_image is not None:
            self.latest_image = None

        self.latest_image = data
        self.latest_image_time = rospy.get_time()

    @staticmethod
    def detect(model, frame):
        """ Return class, contour and bounding box of objects in image per class type. """
        results = model(frame)
        if results and len(results[0]) > 0:
            segmentation_contours_idx = [np.array(seg, dtype=np.int32) for seg in results[0].masks.xy]
            class_ids = np.array(results[0].boxes.cls.cpu(), dtype="int")

            # Get bounding box corners for each detected object
            bounding_box_corners = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in results[0].boxes.xyxy]

            return class_ids, segmentation_contours_idx, bounding_box_corners
        else:
            return None, None, None

    def process_latest_image(self):
        """ Get data from image and publish it to the topic."""

        if self.latest_image is None:
            return

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='passthrough')
        cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)

        classes, segmentations, bounding_box_corners = self.detect(self.model, cv_image)

        if classes is None or segmentations is None:
            self.latest_image = None  # Clear the latest image after processing
            self.latest_image_time = None
            return

        detected_persons = []
        x_positions = []
        nr_persons = 0
        self.batch_nr += 1

        for class_id, seg, box in zip(classes, segmentations, bounding_box_corners):
            x1, y1, x2, y2 = box

            if class_id == self.person_class:
                mask = np.zeros_like(cv_image)
                nr_persons += 1
                cv2.fillPoly(mask, [seg], (255, 255, 255))
                cv_image[mask == 0] = 0
                cropped_image = cv_image[y1:y2, x1:x2]

                image_message = bridge.cv2_to_imgmsg(cropped_image, encoding="passthrough")

                detected_persons.append(image_message)
                x_positions.append((x2-x1)// 2)

        self.latest_image = None  # Clear the latest image after processing
        self.latest_image_time = None

        # Create person_detections msg
        msg = DetectedPerson()
        msg.time = self.latest_image_time
        msg.nr_batch = self.batch_nr
        msg.nr_persons = nr_persons
        msg.detected_persons = detected_persons
        msg. x_positions = x_positions
        self.publisher.publish(msg)

        # for image_message in detected_persons:
        #     self.publisher_debug.publish(image_message)

    def main_loop(self):
        """ Main loop that makes sure only the latest images are processed"""
        while not rospy.is_shutdown():
            # self.msg_callback()
            self.process_latest_image()

            rospy.sleep(0.001)


if __name__ == '__main__':
    try:
        node_pt = PeopleTracker()
        node_pt.main_loop()
    except rospy.exceptions.ROSInterruptException:
        pass
