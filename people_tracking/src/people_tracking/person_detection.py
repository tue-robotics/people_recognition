#!/usr/bin/env python
import sys
import os
import rospkg
import time
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

from sensor_msgs.msg import Image
from people_tracking.msg import DetectedPerson
from std_srvs.srv import Empty, EmptyResponse


NODE_NAME = 'person_detection'
TOPIC_PREFIX = '/hero/'

laptop = sys.argv[1]
name_subscriber_RGB = 'video_frames' if laptop == "True" else '/hero/head_rgbd_sensor/rgb/image_raw'
depth_camera = False if sys.argv[2] == "False" else True
save_data = False if sys.argv[3] == "False" else True


class PersonDetection:
    """ Class for the person detection node."""
    def __init__(self) -> None:
        # Initialize YOLO
        model_path = "~/MEGA/developers/Donal/yolov8n-seg.pt"
        device = "cuda"
        self.model = YOLO(model_path).to(device)
        self.person_class = 0

        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber_rgb = rospy.Subscriber(name_subscriber_RGB, Image, self.image_callback, queue_size=2)
        self.subscriber_depth = rospy.Subscriber('/hero/head_rgbd_sensor/depth_registered/image_raw', Image,
                                                 self.depth_image_callback, queue_size=2)

        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'person_detections', DetectedPerson, queue_size=5)
        self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'debug/segmented_image', Image, queue_size=5)
        self.reset_service = rospy.Service(TOPIC_PREFIX + NODE_NAME + '/reset', Empty, self.reset)

        # Initialize variables
        self.batch_nr = 0
        self.latest_image = None  # To store the most recent image
        self.latest_image_time = None
        self.depth_images = []

    def reset(self, request):
        """ Reset all stored variables in Class to their default values."""
        self.batch_nr = 0
        self.latest_image = None
        self.latest_image_time = None
        self.depth_images = []
        return EmptyResponse()

    def image_callback(self, data: Image) -> None:
        """ Store the latest image with its information."""
        if self.latest_image is not None:
            self.latest_image = None

        self.latest_image = data
        self.latest_image_time = data.header.stamp.secs
        self.batch_nr = data.header.seq

    def depth_image_callback(self, data: Image) -> None:
        """ Store the latest depth image. Only the most recent depth images are stored."""
        while len(self.depth_images) > 50:
            self.depth_images.pop(0)
        self.depth_images.append(data)

    @staticmethod
    def detect(model, frame):
        """ Return class, contour and bounding box of objects in image per class type. """
        results = model(frame, verbose=False)
        if results and len(results[0]) > 0:
            segmentation_contours_idx = [np.array(seg, dtype=np.int32) for seg in results[0].masks.xy]
            class_ids = np.array(results[0].boxes.cls.cpu(), dtype="int")

            # Get bounding box corners for each detected object
            bounding_box_corners = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in results[0].boxes.xyxy]

            return class_ids, segmentation_contours_idx, bounding_box_corners
        else:
            return None, None, None

    @staticmethod
    def key_distance(x: int, middle_image: int = 320) -> int:
        """ Return the x-distance between input x-coordinate and center of the image."""
        return abs(x - middle_image)

    def process_latest_image(self) -> None:
        """ Extract persons with position information from image and publish it to the topic."""

        if self.latest_image is None:
            return
        latest_image = self.latest_image
        latest_image_time = self.latest_image_time
        batch_nr = self.batch_nr

        # Import RGB Image
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(latest_image, desired_encoding='passthrough')
        cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
        # if not laptop:
        #     cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        # Import Depth Image
        if depth_camera:
            cv_depth_image = bridge.imgmsg_to_cv2(self.depth_images[-1], desired_encoding='passthrough')
        else:
            cv_depth_image = None

        if save_data:
            cv2.imwrite(f"{save_path}{batch_nr}.png", cv_image)
            if depth_camera:
                cv2.imwrite(f"{save_path}{batch_nr}_depth.png", cv_depth_image)

        # People detection
        classes, segmentations, bounding_box_corners = self.detect(self.model, cv_image)
        if classes is None or segmentations is None:    # Return if no persons detected in image
            self.latest_image = None
            self.latest_image_time = None
            return

        detected_persons = []
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
                x = int(x1 + ((x2 - x1) / 2))
                y = int(y1 + ((y2 - y1) / 2))

                if depth_camera:
                    # mask_depth = np.zeros_like(cv_depth_image, dtype=np.uint8)
                    # cv2.fillPoly(mask_depth, [seg], (255, 255, 255))
                    #
                    # # Extract the values based on the mask
                    # masked_pixels = cv_depth_image[mask_depth]
                    #
                    # median_color = np.median(masked_pixels)
                    # print("Median color:", median_color)
                    #
                    # cv_depth_image[mask_depth == 0] = 0
                    # depth_cropped = cv_depth_image[y1:y2, x1:x2]
                    # average_color = cv2.mean(cv_depth_image, mask=mask_depth)
                    # rospy.loginfo(f"color {int(average_color[0])}")
                    roi_size = 5
                    roi_x1 = max(0, x - roi_size // 2)
                    roi_y1 = max(0, y - roi_size // 2)
                    roi_x2 = min(cv_depth_image.shape[1], x + roi_size // 2)
                    roi_y2 = min(cv_depth_image.shape[0], y + roi_size // 2)
                    depth_roi = cv_depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
                    z_depth_roi = np.median(depth_roi)

                    z = z_depth_roi
                    rospy.loginfo(f"Z-coord: {z}")
                else:
                    z = 0

                detected_persons.append(image_message)
                x_positions.append(x)
                y_positions.append(y)
                z_positions.append(int(z))

        # Sort entries from smallest to largest  x_distance from the center
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
        msg.z_positions = z_positions
        self.publisher.publish(msg)

        for image_message in detected_persons:  # Publish the segmented images of detected persons
            self.publisher_debug.publish(image_message)

        self.latest_image = None  # Clear the latest image after processing
        self.latest_image_time = None

    def main_loop(self):
        """ Loop to process most recent images. """
        while not rospy.is_shutdown():
            self.process_latest_image()
            rospy.sleep(0.001)


if __name__ == '__main__':
    if save_data:
        try:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path("people_tracking")
            time = time.ctime(time.time())
            save_path = os.path.join(package_path, f'data/{time}_test/')
            os.makedirs(save_path, exist_ok=True)  # Make sure the directory exists
            print(save_path)
        except:
            rospy.loginfo("Failed to make save path")
            pass

    try:
        print(f"Use Depth: {depth_camera}, Camera Source: {name_subscriber_RGB}")
        node_pd = PersonDetection()
        node_pd.main_loop()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Failed to launch Person Detection Node")
        pass
