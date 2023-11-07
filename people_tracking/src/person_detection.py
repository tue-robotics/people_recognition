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
        self.publisher = rospy.Publisher('/hero/segmented_image', Image, queue_size=10)
        # self.subscriber = rospy.Subscriber(name_subscriber_RGB, Image, self.callback, queue_size=1)

        self.publisher2 = rospy.Publisher(TOPIC_PREFIX + 'test_msg', DetectedPerson, queue_size= 10)


        self.latest_image = None  # To store the most recent image
        self.latest_image_time = None

        # Subscribe to the RGB image topic with a callback
        self.subscriber = rospy.Subscriber(name_subscriber_RGB, Image, self.image_callback, queue_size=1)

    # def msg_callback(self):
    #     msg = DetectedPerson()
    #     msg.time = 1
    #     msg.x_position = 5
    #     msg.detected_person = self.latest_image
    #     self.publisher2.publish(msg)
    #     rospy.loginfo("woosssh")

    def image_callback(self, data):
        """ Make sure that only the most recent data will be processed."""
        # Cancel any previously queued image processing tasks
        if self.latest_image is not None:
            self.latest_image = None

        self.latest_image = data
        self.latest_image_time = rospy.get_time()

    @staticmethod
    def detect(model, frame):
        """ Return contour of object in image per class type. """
        results = model(frame)
        if results and len(results[0]) > 0:
            segmentation_contours_idx = [np.array(seg, dtype=np.int32) for seg in results[0].masks.xy]
            class_ids = np.array(results[0].boxes.cls.cpu(), dtype="int")

            # Get bounding box corners (top-left and bottom-right) for each detected object
            bounding_box_corners = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in results[0].boxes.xyxy]

            return class_ids, segmentation_contours_idx, bounding_box_corners
        else:
            return None, None, None

    def process_latest_image(self):
        """ Get data from image and publish it to topic."""

        if self.latest_image is not None:
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='passthrough')
            cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)

            classes, segmentations, bounding_box_corners = self.detect(self.model, cv_image)
            # rospy.loginfo(bounding_box_corners)
            if classes is not None and segmentations is not None: # Check if a person is detected
                # mask = np.zeros_like(cv_image)

                detected_persons = []
                nr_persons = 0
                for class_id, seg, box in zip(classes, segmentations, bounding_box_corners):
                    x1, y1, x2, y2 = box
                    if class_id == self.person_class:
                        detection_image = cv_image.copy()
                        mask = np.zeros_like(detection_image)
                        nr_persons += 1
                        cv2.fillPoly(mask, [seg], (255, 255, 255))
                        detection_image[mask == 0] = 0
                        cropped_image = detection_image[y1:y2, x1:x2]

                        # cv2.rectangle(detection_image, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        image_message = bridge.cv2_to_imgmsg(cropped_image, encoding="passthrough")
                        detected_persons.append(image_message)

                # rospy.loginfo(nr_persons)
                        self.publisher.publish(image_message)

            self.latest_image = None  # Clear the latest image after processing
            self.latest_image_time = None

    def main_loop(self):
        while not rospy.is_shutdown():
            # self.msg_callback()
            self.process_latest_image()

            rospy.sleep(0.1)



if __name__ == '__main__':
    try:
        node_pt = PeopleTracker()
        node_pt.main_loop()
    except rospy.exceptions.ROSInterruptException:
        pass
    # try:
    #     node_pt = PeopleTracker()
    #     rospy.spin()
    # except rospy.exceptions.ROSInterruptException:
    #     pass
