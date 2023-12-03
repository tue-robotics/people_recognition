#!/usr/bin/env python
import sys
import rospy
import cv2
from cv_bridge import CvBridge
import copy

import csv

from typing import List
from collections import namedtuple
from UKFclass import *

# MSGS
from sensor_msgs.msg import Image
from people_tracking.msg import ColourCheckedTarget, ColourTarget, DetectedPerson

# SrvS
from std_srvs.srv import Empty

NODE_NAME = 'people_tracker'
TOPIC_PREFIX = '/hero/'

laptop = sys.argv[1]
name_subscriber_RGB = 'video_frames' if laptop == "True" else '/hero/head_rgbd_sensor/rgb/image_raw'

Persons = namedtuple("Persons",
                     ["nr_batch", "time", "nr_persons", "x_positions", "y_positions", "z_positions", "colour_vectors",
                      "face_detected"])
Target = namedtuple("Target", ["nr_batch", "time", "idx_person", "x", "y", "z"])
Target_hoc = namedtuple("Target_hoc", ["nr_batch", "colour_vector"])


class PeopleTracker:
    def __init__(self) -> None:
        csv_file = open(csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber_hoc = rospy.Subscriber(TOPIC_PREFIX + 'HoC', ColourTarget, self.callback_hoc,
                                               queue_size=2)
        self.subscriber_face = rospy.Subscriber(TOPIC_PREFIX + 'face_detections', ColourCheckedTarget,
                                                self.callback_face, queue_size=2)
        self.subscriber_persons = rospy.Subscriber(TOPIC_PREFIX + 'person_detections', DetectedPerson,
                                                   self.callback_persons, queue_size=2)
        self.subscriber_image_raw = rospy.Subscriber(name_subscriber_RGB, Image, self.get_latest_image, queue_size=1)

        self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'debug/people_tracker', Image, queue_size=2)
        self.rate = rospy.Rate(50)  # 20hz

        # Create a ROS Service Proxys for the reset services
        rospy.wait_for_service(TOPIC_PREFIX + 'HoC/reset')
        self.hoc_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'HoC/reset', Empty)
        rospy.wait_for_service(TOPIC_PREFIX + 'Face/reset')
        self.face_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'Face/reset', Empty)
        rospy.wait_for_service(TOPIC_PREFIX + 'person_detection/reset')
        self.detection_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'person_detection/reset', Empty)

        # Variables
        self.approved_targets = [Target(0, 0, 0, 320, 240, 0)]
        self.approved_targets_hoc = []  # HoC's from approved target (only keep last 10).
        self.time_since_identifiers = None

        self.tracked_plottable = True

        self.detections = []  # All persons detected per frame
        self.new_detections = False

        self.latest_image = None

        self.ukf_from_data = UKF()
        self.ukf_past_data = UKF()

        self.last_timestamp_hoc = None
        self.message_count_hoc = 0
        self.rate_estimate_hoc = 0.0
        self.last_timestamp_face = None
        self.message_count_face = 0
        self.rate_estimate_face = 0.0
        self.last_timestamp_da = None
        self.message_count_da = 0
        self.rate_estimate_da = 0.0

    def reset(self):
        """ Reset all stored variables in Class to their default values."""
        self.approved_targets = [Target(0, 0, 0, 320, 240, 0)]
        self.approved_targets_hoc = []  # HoC's from approved target (only keep last 10).
        self.time_since_identifiers = None

        self.tracked_plottable = True

        self.detections = []  # All persons detected per frame
        self.new_detections = False

        self.latest_image = None

        self.ukf_from_data = UKF()
        self.ukf_past_data = UKF()

    @staticmethod
    def euclidean_distance(point1, point2):
        """ Calculate the Euclidean distance between two points.

        :param point1: A tuple or list representing the coordinates of the first point.
        :param point2: A tuple or list representing the coordinates of the second point.
        :return: The Euclidean distance between the two points.
        """

        if len(point1) != len(point2):
            raise ValueError("Not the same dimensions")

        a = np.array(point1)
        b = np.array(point2)
        return np.linalg.norm(a - b)

    @staticmethod
    def element_exists(lst, element):
        """ Check if element is in list.

        :param lst: List to check element against.
        :param element: Element to check if it is in the list.
        :return: True, index element if in the list, False, None if element not in list
        """
        try:  # Try to find element
            idx = lst.index(element)
            return True, idx
        except ValueError:  # If element is not in the list
            return False, None

    def reset_color_histogram_node(self):
        """ Call the color histogram reset service."""

        try:
            response = self.hoc_reset_proxy()
            rospy.loginfo("Color histogram node reset successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to reset color histogram node: %s", str(e))

    def reset_face_node(self):
        """ Call the face reset service."""
        try:
            response = self.face_reset_proxy()
            rospy.loginfo("Face node reset successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to reset Face node: %s", str(e))

    def reset_detection(self):
        """ Call the people_detection reset service."""
        try:
            response = self.detection_reset_proxy()
            rospy.loginfo("Detection node reset successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to reset detection node: %s", str(e))

    def get_latest_image(self, data: Image) -> None:
        """ Get the most recent frame/image from the camera."""
        self.latest_image = data

    def callback_hoc(self, data: ColourCheckedTarget) -> None:
        """ Add the latest HoC detection to the storage."""
        batch_nr = data.nr_batch
        colour_vectors = [data.colour_vectors[i:i + 32 * 3] for i in range(0, len(data.colour_vectors), 32 * 3)]

        exists, idx = self.element_exists([detection.nr_batch for detection in self.detections], batch_nr)
        if exists:
            nr_batch, time, nr_persons, x_positions, y_positions, z_positions, _, face_detected = self.detections[idx]
            self.detections[idx] = Persons(nr_batch, time, nr_persons, x_positions, y_positions, z_positions,
                                           colour_vectors, face_detected)
            self.new_detections = True

            self.csv_writer.writerow([nr_batch, time, nr_persons,
                                 x_positions, y_positions, z_positions,
                                 colour_vectors, face_detected])
            rospy.loginfo(f"hoc: {nr_batch}")


            current_timestamp = rospy.get_time()
            if self.last_timestamp_hoc is not None:
                time_difference = current_timestamp - self.last_timestamp_hoc
                self.rate_estimate_hoc = 1.0 / time_difference if time_difference > 0 else 0.0

            self.last_timestamp_hoc = current_timestamp
            self.message_count_hoc += 1
        else:
            rospy.loginfo("HoC detection not used")

    def callback_face(self, data: ColourCheckedTarget) -> None:
        """ Add the latest Face detection to the storage."""
        batch_nr = data.batch_nr
        face_detections = data.face_detections
        exists, idx = self.element_exists([detection.nr_batch for detection in self.detections], batch_nr)
        if exists:
            nr_batch, time, nr_persons, x_positions, y_positions, z_positions, colour_vectors, _ = self.detections[idx]
            self.detections[idx] = Persons(nr_batch, time, nr_persons, x_positions, y_positions, z_positions,
                                           colour_vectors, face_detections)
            self.new_detections = True
            self.csv_writer.writerow([nr_batch, time, nr_persons,
                                      x_positions, y_positions, z_positions,
                                      colour_vectors, face_detections])
            rospy.loginfo(f"face: {nr_batch}")

            current_timestamp = rospy.get_time()
            if self.last_timestamp_face is not None:
                time_difference = current_timestamp - self.last_timestamp_face
                self.rate_estimate_face = 1.0 / time_difference if time_difference > 0 else 0.0

            self.last_timestamp_face = current_timestamp
            self.message_count_face += 1
        else:
            rospy.loginfo("Face detection not used")

    def callback_persons(self, data: DetectedPerson) -> None:
        """ Add the latest detected persons from people_detection to the storage."""
        time = data.time
        nr_batch = data.nr_batch
        nr_persons = data.nr_persons
        x_positions = data.x_positions
        y_positions = data.y_positions
        z_positions = data.z_positions
        #
        # if len(self.detections) >= 100:
        #     self.detections.pop(0)
        colour_vectors = [None] * nr_persons
        face_detected = [None] * nr_persons
        self.detections.append(
            Persons(nr_batch, time, nr_persons, x_positions, y_positions, z_positions, colour_vectors, face_detected))
        self.new_detections = True
        self.csv_writer.writerow([nr_batch, time, nr_persons,
                                  x_positions, y_positions, z_positions,
                                  colour_vectors, face_detected])
        rospy.loginfo(f"pos: {nr_batch}")
        current_timestamp = rospy.get_time()
        if self.last_timestamp_da is not None:
            time_difference = current_timestamp - self.last_timestamp_da
            self.rate_estimate_da = 1.0 / time_difference if time_difference > 0 else 0.0

        self.last_timestamp_da = current_timestamp
        self.message_count_da += 1
        # rospy.loginfo([person.nr_batch for person in self.detections])

    def plot_tracker(self):
        """ Plot the trackers on a camera frame and publish it.
        This can be used to visualise all the output from the trackers. [x,y coords] Currently not for depth
        """
        # Convert latest image to cv2 image
        bridge = CvBridge()
        latest_image = self.latest_image
        cv_image = bridge.imgmsg_to_cv2(latest_image, desired_encoding='passthrough')

        if len(self.approved_targets) > 0 and self.tracked_plottable:  # Plot latest approved measurement
            x_approved = self.approved_targets[-1].x
            y_approved = self.approved_targets[-1].y
            cv2.circle(cv_image, (x_approved, y_approved), 5, (255, 0, 0, 50), -1)  # BGR

        # Get location with UKF
        current_time = float(rospy.get_time())
        if self.ukf_from_data.current_time < current_time:  # Get prediction for current time
            ukf_predict = copy.deepcopy(self.ukf_past_data)
            ukf_predict.predict(current_time)
        else:
            ukf_predict = self.ukf_from_data

        x_ukf = int(ukf_predict.kf.x[0])
        y_ukf = int(ukf_predict.kf.x[2])
        cv2.circle(cv_image, (x_ukf, y_ukf), 5, (0, 255, 0, 50), -1)  # BGR

        tracker_image = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        self.publisher_debug.publish(tracker_image)

    def get_distance_hoc(self, hoc):
        """ Compare given hoc to targets last 10 HoC's. Return the smallest distance of hoc. 0 means perfect match."""
        distances = []

        if len(self.approved_targets_hoc) < 1:
            return 0

        test_hocs = self.approved_targets_hoc[-10:]
        for target_hoc in test_hocs:
            distances.append(self.euclidean_distance(target_hoc.colour_vector, hoc))
        return min(distances)

    def track_person(self):
        """ Track the target."""
        rospy.loginfo(f"target: {len(self.approved_targets)} hoc: {len(self.approved_targets_hoc)}")

        # Cycle through measurements
        nr_measurements = 50

        # Clear approved targets to 1
        if len(self.approved_targets) > nr_measurements:
            start_batch = self.approved_targets[-nr_measurements].nr_batch
        else:
            start_batch = self.approved_targets[0].nr_batch

        self.approved_targets = self.remove_outside_batches(self.approved_targets, start_batch, start_batch)

        # Clear hoc to 10 measurements
        self.approved_targets_hoc = self.remove_outside_batches(self.approved_targets_hoc, 0, start_batch)
        while len(self.approved_targets_hoc) > 10:
            self.approved_targets_hoc.pop(0)

        # UKF
        self.ukf_from_data = UKF()
        self.ukf_from_data.update(self.approved_targets[0].time,
                                  [self.approved_targets[0].x, self.approved_targets[0].y, self.approved_targets[0].z])

        self.tracked_plottable = False

        time_since_identifiers = self.time_since_identifiers

        base_idx = len(self.detections) - nr_measurements
        for idx_raw, measurement in enumerate(
                self.detections[-nr_measurements:]):  # Do data association with other data through all detections
            idx = base_idx + idx_raw
            if measurement.nr_persons <= 0:  # Skip measurement if no persons
                continue

            flag_hoc = False
            flag_face = False
            flag_da = False

            if any(x is not None for x in measurement.face_detected):  # There is a face in the measurement
                faces = measurement.face_detected
                flag_face = True

                faces = [0 if value else 1 for value in faces]


            else:  # There is no face data
                faces = [1] * measurement.nr_persons

            if any(x is not None for x in measurement.colour_vectors):  # There is HoC data
                distance_hoc = []  # Get HoC distance to targets
                for person_idx in range(measurement.nr_persons):
                    hoc = measurement.colour_vectors[person_idx]
                    distance_hoc.append(self.get_distance_hoc(hoc))

                # Normalize data
                max_distance_hoc = max(distance_hoc)
                if 0 == max_distance_hoc or len(distance_hoc) <= 1:
                    norm_hoc_distance = [0 for _ in distance_hoc]
                else:
                    norm_hoc_distance = [distance / max_distance_hoc for distance in distance_hoc]

                if any([value < 0.25 for value in distance_hoc]):  # Check if any of the data meets the threshold value
                    flag_hoc = True
            else:  # There is no HoC data
                distance_hoc = []
                norm_hoc_distance = [1] * measurement.nr_persons

            previous_target_coords = (
                self.approved_targets[-1].x, self.approved_targets[-1].y, self.approved_targets[-1].z)
            distance_da = []
            for person_idx in range(measurement.nr_persons):
                position = (measurement.x_positions[person_idx], measurement.y_positions[person_idx],
                            measurement.z_positions[person_idx])
                distance_da.append(self.euclidean_distance(previous_target_coords, position))

            max_da_value = 150
            # Normalize data
            max_distance_da = max(distance_da)
            # rospy.loginfo(f"euclid:{distance_da}")

            if max_distance_da <= 0:
                norm_distance_da = [0 for _ in distance_da]
            else:
                # Adjust values over 200 to have normalized distance 1
                norm_distance_da = [min(value, max_da_value) / max_distance_da if value <= 200 else 1 for value in distance_da]

            if any([value < max_da_value for value in distance_da]):
                flag_da = True

            nr_batch = measurement.nr_batch
            time = measurement.time

            nr_parameters = sum([flag_face, flag_hoc, flag_da])
            current_weight = 2
            weights = [[0.1, 0.2, 0.7],
                       [0.0, 0.3, 0.7],
                       [0.0, 0.0, 1.0]]

            if flag_face:
                weight_face = weights[nr_parameters - 1][current_weight]
                current_weight -= 1
            else:
                weight_face = 0.0

            if flag_hoc:
                weight_hoc = weights[nr_parameters - 1][current_weight]
                current_weight -= 1
            else:
                weight_hoc = 0.0

            if flag_da:
                weight_da = weights[nr_parameters - 1][current_weight]
            else:
                weight_da = 0.0

            combined = [weight_face * faces[person] +
                        weight_hoc * norm_hoc_distance[person] +
                        weight_da * norm_distance_da[person] for person in range(measurement.nr_persons)]
            # rospy.loginfo(f"face: {faces}, hoc: {distance_hoc}, da: {distance_da}, combined: {combined}")
            # rospy.loginfo(f"face: {faces}, norm_hoc: {norm_hoc_distance}, norm_da: {norm_distance_da}")
            idx_target = combined.index(min(combined))
            # rospy.loginfo(f"target: {idx_target}")

            if any([flag_face, flag_hoc, flag_da]):
                x = measurement.x_positions[idx_target]
                y = measurement.y_positions[idx_target]
                z = measurement.z_positions[idx_target]
                self.approved_targets.append(Target(nr_batch, time, idx_target, x, y, z))
                time_since_identifiers = time
                self.ukf_from_data.update(time, [x, y, z])  # UKF

                if flag_hoc:
                    self.approved_targets_hoc.append(Target_hoc(nr_batch, measurement.colour_vectors[idx_target]))
        # rospy.loginfo([(target.nr_batch, target.idx_person) for target in self.approved_targets])

        self.time_since_identifiers = time_since_identifiers
        self.ukf_past_data = copy.deepcopy(self.ukf_from_data)
        self.tracked_plottable = True

    def loop(self):
        """ Loop that repeats itself at self.rate. Can be used to execute methods at given rate. """
        time_old = rospy.get_time()

        while not rospy.is_shutdown():
            # if self.new_detections:  # Do data association with most recent detection.

            if self.latest_image is not None:
                self.plot_tracker()

            # Remove detections older than 500 entries ago
            while len(self.detections) > 500:
                self.detections.pop(0)

            #lOG detection rate
            rospy.loginfo( f"da: {self.rate_estimate_da:.2f} Hz, face: {self.rate_estimate_face:.2f} Hz, hoc: {self.rate_estimate_hoc:.2f} Hz")

            # ToDo Move the picture plotter to different node -> might help with visible lag spikes in tracker
            current_time = rospy.get_time()
            if self.new_detections and current_time - time_old > 0.2:
                time_old = current_time
                # self.track_person()
                self.new_detections = False
                rospy.loginfo(self.detections[-1])



            # Add target lost check (based on time)
            if self.time_since_identifiers is None:
                continue
            if rospy.get_time() - self.time_since_identifiers > 3:
                rospy.loginfo("Target Lost (ID'ers)")

            self.rate.sleep()

    @staticmethod
    def remove_outside_batches(lst: List, start_batch: int = 0, end_batch: int = float("inf")) -> List:
        """ Remove all entries in the given list if the batch is not between start and end batch number.
        :return: list with all the batches removed.
        """
        result = [entry for entry in lst if start_batch <= entry.nr_batch <= end_batch]
        return result

import os
import rospkg
import time

if __name__ == '__main__':
    try:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("people_tracking")
        full_path = os.path.join(package_path, 'data/')

        # Make sure the directory exists
        os.makedirs(full_path, exist_ok=True)
        time = time.ctime(time.time())
        csv_file_path = os.path.join(full_path, f'{time}_test.csv')
    except:
        pass

    try:
        node_pt = PeopleTracker()
        node_pt.loop()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
