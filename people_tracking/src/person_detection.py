#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from people_tracking.msg import DetectedPerson

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
        rospy.init_node('person_detection', anonymous=True)
        self.publisher = rospy.Publisher('/hero/segmented_image', Image, queue_size=10)
        # self.subscriber = rospy.Subscriber(name_subscriber_RGB, Image, self.callback, queue_size=1)
        self.publisher2 = rospy.Publisher('/hero/test_msg', DetectedPerson, queue_size= 10)


        self.latest_image = None  # To store the most recent image
        self.latest_image_time = None

        # Subscribe to the RGB image topic with a callback
        self.subscriber = rospy.Subscriber(name_subscriber_RGB, Image, self.image_callback, queue_size=1)

    def msg_callback(self):
        msg = DetectedPerson()
        msg.time = 1
        msg.x_position = 5
        msg.detected_person = self.latest_image
        self.publisher2.publish(msg)
        rospy.loginfo("woosssh")
    def image_callback(self, data):
        # Cancel any previously queued image processing tasks
        if self.latest_image is not None:
            self.latest_image = None

        self.latest_image = data
        self.latest_image_time = rospy.get_time()

    @staticmethod
    def detect(model, frame):
        """
            Return segemented image per class type.
        """
        results = model(frame)
        result = results[0]
        if len(result) > 0:
            segmentation_contours_idx = [np.array(seg, dtype=np.int32) for seg in result.masks.xy]
            class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
            return class_ids, segmentation_contours_idx
        else:
             return None, None

    def process_latest_image(self):
        if self.latest_image is not None:
            # rospy.loginfo("Processing the latest image")
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='passthrough')
            cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)

            classes, segmentations = self.detect(self.model, cv_image)
            if classes is not None and segmentations is not None: # Check if a person is detected
                mask = np.zeros_like(cv_image)

                for class_id, seg in zip(classes, segmentations):
                    if class_id == self.person_class:
                        cv2.fillPoly(mask, [seg], (255, 255, 255))

                cv_image[mask == 0] = 0

                image_message = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")

                self.publisher.publish(image_message)
            rospy.loginfo(self.latest_image_time)
            self.latest_image = None  # Clear the latest image after processing
            self.latest_image_time = None

    def main_loop(self):
        while not rospy.is_shutdown():
            self.msg_callback()
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
