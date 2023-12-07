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

save_data = False if sys.argv[3] == "False" else True


Persons = namedtuple("Persons",
                     ["nr_batch", "time", "nr_persons", "x_positions", "y_positions", "z_positions", "colour_vectors",
                      "face_detected"])
Target = namedtuple("Target", ["nr_batch", "time", "idx_person", "x", "y", "z", "colour_vector", "valid_measurement"])
Target_hoc = namedtuple("Target_hoc", ["nr_batch", "colour_vector"])


class PeopleTracker:
    def __init__(self) -> None:
        if save_data:
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
        # self.publisher = rospy.Publisher(TOPIC_PREFIX + 'tracker', Image, queue_size=2)
        self.rate = rospy.Rate(50)  # 20hz

        # Create a ROS Service Proxys for the reset services
        rospy.wait_for_service(TOPIC_PREFIX + 'HoC/reset')
        self.hoc_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'HoC/reset', Empty)
        rospy.wait_for_service(TOPIC_PREFIX + 'Face/reset')
        self.face_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'Face/reset', Empty)
        rospy.wait_for_service(TOPIC_PREFIX + 'person_detection/reset')
        self.detection_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'person_detection/reset', Empty)

        # Variables
        self.approved_targets = [Target(0, 0, 0, 320, 240, 0, None, True)]
        self.approved_targets_hoc = []  # HoC's from approved target (only keep last 10).
        self.time_since_identifiers = None

        self.tracked_plottable = True

        self.detections = []  # All persons detected per frame
        self.new_detections = False

        self.latest_image = None

        self.ukf_from_data = UKF()
        # self.ukf_past_data = UKF()

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
        # self.ukf_past_data = UKF()

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
            if save_data:
                self.csv_writer.writerow([nr_batch, time, nr_persons,
                                     x_positions, y_positions, z_positions,
                                     colour_vectors, face_detected])
            # rospy.loginfo(f"hoc: {nr_batch}")


            current_timestamp = rospy.get_time()
            if self.last_timestamp_hoc is not None:
                time_difference = current_timestamp - self.last_timestamp_hoc
                self.rate_estimate_hoc = 1.0 / time_difference if time_difference > 0 else 0.0

            self.last_timestamp_hoc = current_timestamp
            self.message_count_hoc += 1

            self.update_target(nr_batch)

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
            if save_data:
                self.csv_writer.writerow([nr_batch, time, nr_persons,
                                          x_positions, y_positions, z_positions,
                                          colour_vectors, face_detections])
            # rospy.loginfo(f"face: {nr_batch}")

            current_timestamp = rospy.get_time()
            if self.last_timestamp_face is not None:
                time_difference = current_timestamp - self.last_timestamp_face
                self.rate_estimate_face = 1.0 / time_difference if time_difference > 0 else 0.0

            self.last_timestamp_face = current_timestamp
            self.message_count_face += 1

            self.update_target(nr_batch)
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
        if save_data:
            self.csv_writer.writerow([nr_batch, time, nr_persons,
                                      x_positions, y_positions, z_positions,
                                      colour_vectors, face_detected])
        # rospy.loginfo(f"pos: {nr_batch}")
        current_timestamp = rospy.get_time()
        if self.last_timestamp_da is not None:
            time_difference = current_timestamp - self.last_timestamp_da
            self.rate_estimate_da = 1.0 / time_difference if time_difference > 0 else 0.0

        self.last_timestamp_da = current_timestamp
        self.message_count_da += 1

        self.update_target(nr_batch)
        # rospy.loginfo([person.nr_batch for person in self.detections])

    def check_face_data(self, detection):
        """
        :param detection: the detection to check face from
        :return: flag_face, faces (0 = correct face, 1 = wrong face)
        """
        if any(x is not None for x in detection.face_detected):  # There is a face in the detection
            faces = detection.face_detected
            flag_face = True
            faces = [0 if value else 2 for value in faces]
        else:  # There is no face data
            flag_face = False
            faces = [1] * detection.nr_persons

        return flag_face, faces

    def get_distance_hocs(self, hoc_check, hocs_existing):
        """
        :param hoc_check: Hoc to check agains list of hocs
        :param hocs_existing: hocs to calculate distance to hoc_check from
        :return: minimum distances from input list
        """
        distances = [self.euclidean_distance(hoc_check, hoc) for hoc in hocs_existing]
        return min(distances)

    def check_hoc_data(self, detection, tracked_hocs):
        """
        :param detection: the detection to check hocs from
        :param tracked_hocs: hocs of previous detections
        :return: flag_hoc, norm_hoc (0 = perfect match, 1 = worst match)
        """
        HOC_THRESHOLD = 0.25

        if len(tracked_hocs) <= 0:  # If there is no existing HOC data make sure hoc is not taken into account
            return False, [1] * detection.nr_persons

        if any(x is not None for x in detection.colour_vectors):  # There is HoC data
            flag_hoc = True

            distance_hoc = [self.get_distance_hocs(detection.colour_vectors[person_idx], tracked_hocs) for person_idx in range(detection.nr_persons)]

            if any([value < HOC_THRESHOLD for value in distance_hoc]):  # Check if any of the data meets the threshold
                # Normalize data
                max_distance_hoc = max([distance for distance in distance_hoc if distance < HOC_THRESHOLD]) # get max distance without invalid entries
                if 0 == max_distance_hoc or len(distance_hoc) <= 1:
                    norm_hoc = [0 for _ in distance_hoc]
                else:
                    norm_hoc = [distance / max_distance_hoc if distance < HOC_THRESHOLD else 2 for distance in distance_hoc]

            else:   # All values are invalid, thus max normalized distance
                norm_hoc = [2] * detection.nr_persons

        else:  # There is no HoC data
            flag_hoc = False
            norm_hoc = [1] * detection.nr_persons

        return flag_hoc, norm_hoc


    def check_da_data(self, new_detection, previous_da_detection):
        """
        :param new_detection: detection to calculate the distance per measurement
        :param previous_da_detection: measurement to calculate the distance from (aka the previous known target location).
        :return:
        """
        DA_THRESHOLD = 150

        flag_da = True
        previous_target_coords = (previous_da_detection.x, previous_da_detection.y, previous_da_detection.z)

        coords_detections = [(new_detection.x_positions[person_idx], new_detection.y_positions[person_idx], new_detection.z_positions[person_idx]) for person_idx in range(new_detection.nr_persons)]
        distance_da = [self.euclidean_distance(previous_target_coords, detection) for detection in coords_detections]

        if any([value < DA_THRESHOLD for value in distance_da]):

            # Normalize data
            max_distance = max([distance for distance in distance_da if distance < DA_THRESHOLD])  # get max distance without invalid entries
            if 0 == max_distance or len(distance_da) <= 1:
                norm_da = [0 for _ in distance_da]
            else:
                norm_da = [distance / max_distance if distance < DA_THRESHOLD else 2 for distance in distance_da]

        else:       # All data is invalid
            norm_da = [2] * new_detection.nr_persons

        return flag_da, norm_da

    def get_target_value(self, new_detection, tracked_hocs, previous_da_detection, flag_target_lost):
        """ Calculate the data association between two detection. Return the idx of the target"""

        if new_detection.nr_persons < 1:
            return None, False

        flag_face, faces = self.check_face_data(new_detection)
        # print(f"flag_face {flag_face}, {faces}")

        flag_hoc, norm_hoc = self.check_hoc_data(new_detection, tracked_hocs)
        # print(f"flag_hoc {flag_hoc}, {norm_hoc}")

        flag_da, norm_da = self.check_da_data(new_detection, previous_da_detection)
        # print(f"flag_da {flag_da}, {norm_da}")

        weight_face, weight_hoc, weight_da = self.get_weights(flag_face, flag_hoc, flag_da, flag_target_lost)
        # print(f"{new_detection.nr_batch}, flags: {flag_face, flag_hoc, flag_da}, weights: {weight_face, weight_hoc, weight_da}")
        combined = [weight_face * faces[person] + weight_hoc * norm_hoc[person] + weight_da * norm_da[person]
                    for person in range(new_detection.nr_persons)]

        idx_target = combined.index(min(combined))
        valid = True if min(combined) <= 1 else False   # combined is larger than 1 if either to many of the targets have invalid measurements or the most important one is invalid.
        return idx_target, valid

    def add_approved_target(self, measurement, idx_target, valid):
        """ Add approved target with data to list."""

        nr_batch = measurement.nr_batch
        time = measurement.time

        if idx_target is None:
            x = self.approved_targets[-1].x
            y = self.approved_targets[-1].y
            z = self.approved_targets[-1].z
            colour_vector = None

        else:
            x = measurement.x_positions[idx_target]
            y = measurement.y_positions[idx_target]
            z = measurement.z_positions[idx_target]
            colour_vector = measurement.colour_vectors[idx_target]
            # print(f"colour {colour_vector}")
        self.approved_targets.append(Target(nr_batch, time, idx_target, x, y, z, colour_vector, valid))

        # print(valid)
        if valid and self.ukf_from_data.current_time <= time:
            self.ukf_from_data.update(time, [x, y, z])
            # print("update ukf")
        # print(f"approved {nr_batch},  {valid}")

    def update_approved_target(self, idx_target, idx_tracked, measurement, valid):
        """

        :param idx_target: the target idx in the measurement
        :param idx_tracked: idx of the tracked target (e.g. the idx in self.tracked_targets
        :param measurement: the new detection to update tracker with
        :return: None
        """
        if idx_target is None:
            return

        if any(x is None for x in measurement.colour_vectors):
            return

        nr_batch = measurement.nr_batch
        time = measurement.time
        x = measurement.x_positions[idx_target]
        y = measurement.y_positions[idx_target]
        z = measurement.z_positions[idx_target]
        colour_vector = measurement.colour_vectors[idx_target]

        self.approved_targets[idx_tracked] = Target(nr_batch, time, idx_target, x, y, z, colour_vector, valid)
        # print(f"updated {nr_batch},  {valid}")

    def get_tracked_hocs(self, idx_tracked=None):
        # Get 5 previous hoc measurements from track
        hoc_idx = 1
        tracked_hocs = []

        if idx_tracked is not None:
            while len(tracked_hocs) < 10 and hoc_idx < 60 and hoc_idx < len(self.approved_targets[:idx_tracked]):
                if self.approved_targets[idx_tracked - hoc_idx].colour_vector is not None and \
                        self.approved_targets[idx_tracked - hoc_idx].valid_measurement:
                    tracked_hocs.append(self.approved_targets[idx_tracked - hoc_idx].colour_vector)
                hoc_idx += 1
        else:
            while len(tracked_hocs) < 10 and hoc_idx < 60 and hoc_idx < len(self.approved_targets):
                if self.approved_targets[-hoc_idx].colour_vector is not None and \
                        self.approved_targets[-hoc_idx].valid_measurement:
                    tracked_hocs.append(self.approved_targets[-hoc_idx].colour_vector)
                hoc_idx += 1

        return tracked_hocs

    def update_target(self, from_batch):
        """ Update the self.approved_targets from batch."""

        exists_detection, idx_detection = self.element_exists([detection.nr_batch for detection in self.detections], from_batch)
        if not exists_detection:  # Make sure existing batch number in detections
            return

        exist_tracked, idx_tracked = self.element_exists([detection.nr_batch for detection in self.approved_targets], from_batch)

        if exist_tracked:   # Check new data with existing track.
            idx_compare = idx_tracked-1
            while not self.approved_targets[idx_compare].valid_measurement:
                idx_compare -= 1

            tracked_hocs = self.get_tracked_hocs(idx_tracked)
            # print(f"tracked hocs exists: {tracked_hocs}")
            idx_target, valid = self.get_target_value(self.detections[idx_detection], tracked_hocs,
                                                      self.approved_targets[idx_compare],
                                                      self.approved_targets[idx_compare].valid_measurement)

            if self.approved_targets[idx_tracked].idx_person == idx_target:
                self.update_approved_target(idx_target, idx_tracked, self.detections[idx_detection], valid)
                # print(f"correct {valid}")
                return
            else:
                # print(f"dummu {self.approved_targets[idx_tracked].idx_person} {idx_target}")

                self.approved_targets = self.approved_targets[:idx_compare]
                self.ukf_from_data = UKF()


                while idx_detection < len(self.detections)-1:
                    tracked_hocs = self.get_tracked_hocs()
                    # print(f"tracked hocs new: {tracked_hocs}")

                    idx_target, valid = self.get_target_value(self.detections[idx_detection], tracked_hocs,
                                                              self.approved_targets[-1],
                                                              self.approved_targets[-1].valid_measurement)
                    self.add_approved_target(self.detections[idx_detection], idx_target, valid)

                    idx_detection += 1

                return



        if self.approved_targets[-1].nr_batch < from_batch:  # Add single data association step to the end of target list

            # Get 5 previous hoc measurements from track
            tracked_hocs = self.get_tracked_hocs()
            # print(f"tracked hocs new: {tracked_hocs}")

            idx_target, valid = self.get_target_value(self.detections[idx_detection], tracked_hocs, self.approved_targets[-1], self.approved_targets[-1].valid_measurement)
            self.add_approved_target(self.detections[idx_detection], idx_target, valid)

            return

        print("TUMMMMMM")
        # else: # totaly new data that can be placed somewhere between already associated data, find place, do associaiton 1 back to this, this to next, if match in next return else redo everything -> For now do nothing with it





    @staticmethod
    def get_weights(flag_face, flag_hoc, flag_da, valid):
        """ Get the correct weights for the DA.

        :return: weight_face, weight_hoc, weight_da
        """
        # print(f"weight:{sum([flag_face,flag_hoc,flag_da])} {valid}")
        if sum([flag_face, flag_hoc, flag_da]) <= 0:
            return 0.0, 0.0, 0.0

        if not valid:    # Use different weights if target was lost in previous steps
            weight_da = 0.0

            nr_parameters = sum([flag_face, flag_hoc])
            weights = [[1.0, 1.0],      # 1 parameter
                       [0.9, 0.1]]      # 2 parameters

            if flag_face:
                weight_face = weights[nr_parameters-1][0]
            else:
                weight_face = 0.0

            if flag_hoc:
                weight_hoc = weights[nr_parameters - 1][1]
            else:
                weight_hoc = 0.0

        else:
            nr_parameters = sum([flag_face, flag_hoc, flag_da])  # How many measurement types available
            current_weight = 2
            weights = [[0.0, 0.0, 1.0],     # 1 parameter
                       [0.0, 0.2, 0.8],     # 2 parameters
                       [0.1, 0.2, 0.7]]     # 3 parameters

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

        return weight_face, weight_hoc, weight_da
    def plot_tracker(self):
        """ Plot the trackers on a camera frame and publish it.
        This can be used to visualise all the output from the trackers. [x,y coords] Currently not for depth data
        """
        # Convert latest image to cv2 image
        bridge = CvBridge()
        latest_image = self.latest_image
        cv_image = bridge.imgmsg_to_cv2(latest_image, desired_encoding='passthrough')

        if len(self.approved_targets) > 0 and self.tracked_plottable:  # Plot latest approved measurement
            x_approved = self.approved_targets[-1].x
            y_approved = self.approved_targets[-1].y
            cv2.circle(cv_image, (x_approved, y_approved), 5, (0, 0, 255, 50), -1)  # BGR

        # Get location with UKF
        current_time = float(rospy.get_time())
        ukf_predict = copy.deepcopy(self.ukf_from_data)
        if ukf_predict.current_time < current_time:  # Get prediction for current time
            ukf_predict.predict(current_time)

        x_ukf = int(ukf_predict.kf.x[0])
        y_ukf = int(ukf_predict.kf.x[2])
        cv2.circle(cv_image, (x_ukf, y_ukf), 5, (0, 255, 0, 50), -1)  # BGR

        tracker_image = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        self.publisher_debug.publish(tracker_image)
    #
    # def get_distance_hoc(self, hoc):
    #     """ Compare given hoc to targets last 10 HoC's. Return the smallest distance of hoc. 0 means perfect match."""
    #     distances = []
    #
    #     if len(self.approved_targets_hoc) < 1:
    #         return 0
    #
    #     test_hocs = self.approved_targets_hoc[-10:]
    #     for target_hoc in test_hocs:
    #         distances.append(self.euclidean_distance(target_hoc.colour_vector, hoc))
    #     return min(distances)
    #
    # def track_person(self):
    #     """ Track the target."""
    #     rospy.loginfo(f"target: {len(self.approved_targets)} hoc: {len(self.approved_targets_hoc)}")
    #
    #     # Cycle through measurements
    #     nr_measurements = 50
    #
    #     # Clear approved targets to 1
    #     if len(self.approved_targets) > nr_measurements:
    #         start_batch = self.approved_targets[-nr_measurements].nr_batch
    #     else:
    #         start_batch = self.approved_targets[0].nr_batch
    #
    #     self.approved_targets = self.remove_outside_batches(self.approved_targets, start_batch, start_batch)
    #
    #     # Clear hoc to 10 measurements
    #     self.approved_targets_hoc = self.remove_outside_batches(self.approved_targets_hoc, 0, start_batch)
    #     while len(self.approved_targets_hoc) > 10:
    #         self.approved_targets_hoc.pop(0)
    #
    #     # UKF
    #     self.ukf_from_data = UKF()
    #     self.ukf_from_data.update(self.approved_targets[0].time,
    #                               [self.approved_targets[0].x, self.approved_targets[0].y, self.approved_targets[0].z])
    #
    #     self.tracked_plottable = False
    #
    #     time_since_identifiers = self.time_since_identifiers
    #
    #     base_idx = len(self.detections) - nr_measurements
    #     for idx_raw, measurement in enumerate(
    #             self.detections[-nr_measurements:]):  # Do data association with other data through all detections
    #         idx = base_idx + idx_raw
    #         if measurement.nr_persons <= 0:  # Skip measurement if no persons
    #             continue
    #
    #         flag_hoc = False
    #         flag_face = False
    #         flag_da = False
    #
    #         if any(x is not None for x in measurement.face_detected):  # There is a face in the measurement
    #             faces = measurement.face_detected
    #             flag_face = True
    #
    #             faces = [0 if value else 1 for value in faces]
    #
    #
    #         else:  # There is no face data
    #             faces = [1] * measurement.nr_persons
    #
    #         if any(x is not None for x in measurement.colour_vectors):  # There is HoC data
    #             distance_hoc = []  # Get HoC distance to targets
    #             for person_idx in range(measurement.nr_persons):
    #                 hoc = measurement.colour_vectors[person_idx]
    #                 distance_hoc.append(self.get_distance_hoc(hoc))
    #
    #             # Normalize data
    #             max_distance_hoc = max(distance_hoc)
    #             if 0 == max_distance_hoc or len(distance_hoc) <= 1:
    #                 norm_hoc_distance = [0 for _ in distance_hoc]
    #             else:
    #                 norm_hoc_distance = [distance / max_distance_hoc for distance in distance_hoc]
    #
    #             if any([value < 0.25 for value in distance_hoc]):  # Check if any of the data meets the threshold value
    #                 flag_hoc = True
    #         else:  # There is no HoC data
    #             distance_hoc = []
    #             norm_hoc_distance = [1] * measurement.nr_persons
    #
    #         previous_target_coords = (
    #             self.approved_targets[-1].x, self.approved_targets[-1].y, self.approved_targets[-1].z)
    #         distance_da = []
    #         for person_idx in range(measurement.nr_persons):
    #             position = (measurement.x_positions[person_idx], measurement.y_positions[person_idx],
    #                         measurement.z_positions[person_idx])
    #             distance_da.append(self.euclidean_distance(previous_target_coords, position))
    #
    #         max_da_value = 150
    #         # Normalize data
    #         max_distance_da = max(distance_da)
    #         # rospy.loginfo(f"euclid:{distance_da}")
    #
    #         if max_distance_da <= 0:
    #             norm_distance_da = [0 for _ in distance_da]
    #         else:
    #             # Adjust values over 200 to have normalized distance 1
    #             norm_distance_da = [min(value, max_da_value) / max_distance_da if value <= 200 else 1 for value in distance_da]
    #
    #         if any([value < max_da_value for value in distance_da]):
    #             flag_da = True
    #
    #         nr_batch = measurement.nr_batch
    #         time = measurement.time
    #
    #         nr_parameters = sum([flag_face, flag_hoc, flag_da])
    #         current_weight = 2
    #         weights = [[0.1, 0.2, 0.7],
    #                    [0.0, 0.3, 0.7],
    #                    [0.0, 0.0, 1.0]]
    #
    #         if flag_face:
    #             weight_face = weights[nr_parameters - 1][current_weight]
    #             current_weight -= 1
    #         else:
    #             weight_face = 0.0
    #
    #         if flag_hoc:
    #             weight_hoc = weights[nr_parameters - 1][current_weight]
    #             current_weight -= 1
    #         else:
    #             weight_hoc = 0.0
    #
    #         if flag_da:
    #             weight_da = weights[nr_parameters - 1][current_weight]
    #         else:
    #             weight_da = 0.0
    #
    #         combined = [weight_face * faces[person] +
    #                     weight_hoc * norm_hoc_distance[person] +
    #                     weight_da * norm_distance_da[person] for person in range(measurement.nr_persons)]
    #         # rospy.loginfo(f"face: {faces}, hoc: {distance_hoc}, da: {distance_da}, combined: {combined}")
    #         # rospy.loginfo(f"face: {faces}, norm_hoc: {norm_hoc_distance}, norm_da: {norm_distance_da}")
    #         idx_target = combined.index(min(combined))
    #         # rospy.loginfo(f"target: {idx_target}")
    #
    #         if any([flag_face, flag_hoc, flag_da]):
    #             x = measurement.x_positions[idx_target]
    #             y = measurement.y_positions[idx_target]
    #             z = measurement.z_positions[idx_target]
    #             self.approved_targets.append(Target(nr_batch, time, idx_target, x, y, z))
    #             time_since_identifiers = time
    #             self.ukf_from_data.update(time, [x, y, z])  # UKF
    #
    #             if flag_hoc:
    #                 self.approved_targets_hoc.append(Target_hoc(nr_batch, measurement.colour_vectors[idx_target]))
    #     # rospy.loginfo([(target.nr_batch, target.idx_person) for target in self.approved_targets])
    #
    #     self.time_since_identifiers = time_since_identifiers
    #     self.ukf_past_data = copy.deepcopy(self.ukf_from_data)
    #     self.tracked_plottable = True

    def loop(self):
        """ Loop that repeats itself at self.rate. Can be used to execute methods at given rate. """
        time_old = rospy.get_time()

        while not rospy.is_shutdown():
            # if self.new_detections:  # Do data association with most recent detection.

            if self.latest_image is not None:
                self.plot_tracker()

            # # Remove detections older than 500 entries ago
            # while len(self.detections) > 500:
            #     self.detections.pop(0)
            #
            # #lOG detection rate
            current_time = rospy.get_time()
            if len(self.approved_targets) > 0:
                val_idx = 1
                validity = False
                while not validity and val_idx < 15 and val_idx < len(self.approved_targets)-1:
                    validity = self.approved_targets[-val_idx].valid_measurement
                    val_idx += 1

            if current_time - self.approved_targets[-val_idx].time > 3:
                rospy.loginfo("Target Lost")

            if current_time - time_old > 0.1:
                rospy.loginfo( f"da: {self.rate_estimate_da:.2f} Hz, face: {self.rate_estimate_face:.2f} Hz, hoc: {self.rate_estimate_hoc:.2f} Hz")
                time_old = current_time
            #
            # # ToDo Move the picture plotter to different node -> might help with visible lag spikes in tracker
            # current_time = rospy.get_time()
            # if self.new_detections and current_time - time_old > 0.2:
            #     time_old = current_time
            #     self.track_person()
            #     self.new_detections = False
            #     rospy.loginfo(self.detections[-1])
            #
            #
            #
            # # Add target lost check (based on time)
            # if self.time_since_identifiers is None:
            #     continue
            # if rospy.get_time() - self.time_since_identifiers > 3:
            #     rospy.loginfo("Target Lost (ID'ers)")

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
    if save_data:
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
