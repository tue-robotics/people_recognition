#!/usr/bin/env python
import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

# MSGS
from sensor_msgs.msg import Image
from people_tracking.msg import DetectedPerson

from std_srvs.srv import Empty, EmptyResponse
from people_tracking.srv import Depth

NODE_NAME = 'person_detection'
TOPIC_PREFIX = '/hero/'

laptop = sys.argv[1]
name_subscriber_RGB = 'video_frames' if laptop == "True" else '/hero/head_rgbd_sensor/rgb/image_raw'


class PersonDetector:
    def __init__(self) -> None:
        # Initialize YOLO
        model_path = "~/MEGA/developers/Donal/yolov8n-seg.pt"
        device = "cuda"
        self.model = YOLO(model_path).to(device)
        self.person_class = 0

        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber = rospy.Subscriber(name_subscriber_RGB, Image, self.image_callback, queue_size=1)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'person_detections', DetectedPerson, queue_size=5)
        self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'debug/segmented_image', Image, queue_size=5)
        self.reset_service = rospy.Service(TOPIC_PREFIX + NODE_NAME + '/reset', Empty, self.reset)

        # Initialize variables
        self.batch_nr = 0
        self.latest_image = None  # To store the most recent image
        self.latest_image_time = None


        # depth
        rospy.wait_for_service(TOPIC_PREFIX + 'depth/depth_data')
        self.depth_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'depth/depth_data', Depth)

    def reset(self, request):
        """ Reset all stored variables in Class to their default values."""
        self.batch_nr = 0
        self.latest_image = None
        self.latest_image_time = None
        return EmptyResponse()

    def request_depth_image(self, time_stamp):

        try:
            if time_stamp is not None:
                response = self.depth_proxy(int(time_stamp))
                return response
                # rospy.loginfo(f"Depth: {response}")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to get depth: %s", str(e))


    def image_callback(self, data):
        """ Make sure that only the most recent data will be processed."""
        if self.latest_image is not None:
            self.latest_image = None

        self.latest_image = data
        # rospy.loginfo("%s, %s",data.header.seq, data.header.stamp.secs)
        self.latest_image_time = data.header.stamp.secs#float(rospy.get_time())
        self.batch_nr = data.header.seq
        # rospy.loginfo("rgb: %s t: %s",data.header.seq, data.header.stamp.secs)

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

    @staticmethod
    def key_distance(x):
        """ Calculate x-distance between input and center image."""
        x_middle_image = 320
        return abs(x - x_middle_image)

    def process_latest_image(self):
        """ Get data from image and publish it to the topic."""

        if self.latest_image is None:
            return
        latest_image = self.latest_image
        latest_image_time = self.latest_image_time
        batch_nr = self.batch_nr
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(latest_image, desired_encoding='passthrough')
        cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)

        depth_image = self.request_depth_image(self.latest_image_time).image
        # rospy.loginfo(type(depth_image))
        cv_depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        cv_depth_image = cv2.GaussianBlur(cv_depth_image, (5, 5), 0)

        classes, segmentations, bounding_box_corners = self.detect(self.model, cv_image)

        if classes is None or segmentations is None:
            self.latest_image = None  # Clear the latest image after processing
            self.latest_image_time = None
            return

        detected_persons = []
        depth_detected = []
        x_positions = []
        y_positions = []
        z_positions = []
        nr_persons = 0

        for class_id, seg, box in zip(classes, segmentations, bounding_box_corners):
            x1, y1, x2, y2 = box

            if class_id == self.person_class:
                mask = np.zeros_like(cv_image)
                nr_persons += 1
                cv2.fillPoly(mask, [seg], (255, 255, 255))
                cv_image[mask == 0] = 0
                cropped_image = cv_image[y1:y2, x1:x2]
                image_message = bridge.cv2_to_imgmsg(cropped_image, encoding="passthrough")

                mask_depth = np.zeros_like(cv_depth_image, dtype=np.uint8)
                cv2.fillPoly(mask_depth, [seg], (255, 255, 255))
                average_color = cv2.mean(cv_depth_image, mask=mask_depth)
                cv_depth_image[mask_depth == 0] = 0
                depth_cropped = cv_depth_image[y1:y2, x1:x2]
                image_message_depth = bridge.cv2_to_imgmsg(depth_cropped, encoding="passthrough")
                depth_detected.append(image_message_depth)

                # rospy.loginfo(f"color {int(average_color[0])}")
                z_positions.append(int(average_color[0]))

                detected_persons.append(image_message)
                x_positions.append(int(x1 + ((x2 - x1) / 2)))
                y_positions.append(int(y1 + ((y2 - y1) / 2)))

        # Sort entries from x_distance from the center, makes sure that the first round correct data is selected
        sorted_idx = sorted(range(len(x_positions)), key=lambda k: self.key_distance(x_positions[k]))
        detected_persons = [detected_persons[i] for i in sorted_idx]
        x_positions = [x_positions[i] for i in sorted_idx]
        y_positions = [y_positions[i] for i in sorted_idx]
        z_positions = [z_positions[i] for i in sorted_idx]

        # Create and Publish person_detections msg
        msg = DetectedPerson()
        msg.time = float(latest_image_time)
        msg.nr_batch = batch_nr
        msg.nr_persons = nr_persons
        msg.detected_persons = detected_persons
        msg.x_positions = x_positions
        msg.y_positions = y_positions
        msg.z_positions = z_positions#[0] * nr_persons  # Temporary, will be replaced with depth data
        self.publisher.publish(msg)

        self.latest_image = None  # Clear the latest image after processing
        self.latest_image_time = None

        for image_message in depth_detected:
            self.publisher_debug.publish(image_message)
        # for image_message in detected_persons:
        #     self.publisher_debug.publish(image_message)

    def main_loop(self):
        """ Main loop that makes sure only the latest images are processed. """
        while not rospy.is_shutdown():
            self.process_latest_image()
            rospy.sleep(0.001)


if __name__ == '__main__':
    try:
        node_pd = PersonDetector()
        node_pd.main_loop()
    except rospy.exceptions.ROSInterruptException:
        pass
