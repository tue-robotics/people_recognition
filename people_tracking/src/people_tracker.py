#!/usr/bin/env python
import sys
import rospy
import cv2
import math
from cv_bridge import CvBridge
import copy

from typing import List, Union

from UKFclass import *

# MSGS
from sensor_msgs.msg import Image
from people_tracking.msg import ColourCheckedTarget, ColourTarget
from people_tracking.msg import DetectedPerson

# SrvS
from std_srvs.srv import Empty

NODE_NAME = 'people_tracker'
TOPIC_PREFIX = '/hero/'

laptop = sys.argv[1]
name_subscriber_RGB = 'video_frames' if laptop == "True" else '/hero/head_rgbd_sensor/rgb/image_raw'

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

from collections import namedtuple

# Detection = namedtuple("Detection", ["batch_nr", "idx_person", "time", "x", "y", "z"])
Persons = namedtuple("Persons", ["nr_batch", "time", "nr_persons", "x_positions", "y_positions", "z_positions", "colour_vectors", "face_detected"])



class PeopleTracker:
    def __init__(self) -> None:

        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber_hoc = rospy.Subscriber(TOPIC_PREFIX + 'HoC', ColourTarget, self.callback_hoc,
                                               queue_size=5)
        self.subscriber_face = rospy.Subscriber(TOPIC_PREFIX + 'face_detections', ColourCheckedTarget,
                                                self.callback_face, queue_size=5)
        self.subscriber_persons = rospy.Subscriber(TOPIC_PREFIX + 'person_detections', DetectedPerson,
                                                   self.callback_persons, queue_size=5)
        self.subscriber_image_raw = rospy.Subscriber(name_subscriber_RGB, Image, self.get_latest_image, queue_size=1)

        self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'debug/people_tracker', Image, queue_size=10)
        self.rate = rospy.Rate(20)  # 20hz

        # Create a ROS Service Proxys for the reset services
        rospy.wait_for_service(TOPIC_PREFIX + 'HoC/reset')
        self.hoc_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'HoC/reset', Empty)
        rospy.wait_for_service(TOPIC_PREFIX + 'Face/reset')
        self.face_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'Face/reset', Empty)
        rospy.wait_for_service(TOPIC_PREFIX + 'person_detection/reset')
        self.detection_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'person_detection/reset', Empty)

        # Variables
        self.hoc_detections = []  # latest detected persons with hoc
        self.new_hoc_detection = False
        self.time_hoc_detection_sec = None

        self.face_detections = []  # latest detected persons with face
        self.new_face_detection = False
        self.time_face_detection_sec = None

        self.detections = []  # All persons detected per frame
        self.new_detections = False

        self.latest_image = None

        self.ukf_hoc = UKF()
        self.data_hoc = []

        self.ukf_face = UKF()
        self.data_face = []  # list of data added to confirmed and not to face

        self.ukf_data_association = UKF()   # UKF with the data association
        self.data_data_association = []

        self.ukf_confirmed = UKF()

        self.HoC_detections = []

    def reset(self):
        """ Reset all stored variables in Class to their default values."""
        self.hoc_detections = []  # latest detected persons with hoc
        self.new_hoc_detection = False
        self.time_hoc_detection_sec = None

        self.face_detections = []  # latest detected persons with face
        self.new_face_detection = False
        self.time_face_detection_sec = None

        self.detections = []  # All persons detected per frame

        self.latest_image = None

        self.ukf_hoc = UKF()
        self.data_hoc = []

        self.ukf_face = UKF()
        self.data_face = []  # list of data added to confirmed and not to face

        self.ukf_confirmed = UKF()

        self.ukf_data_association = UKF()
        self.data_data_association = []

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

    def callback_hoc(self, data: ColourCheckedTarget, amount_detections_stored: int = 100) -> None:
        """ Add the latest HoC detection to the storage."""
        batch_nr = data.nr_batch
        colour_vectors = [data.colour_vectors[i:i + 32*3] for i in range(0, len(data.colour_vectors), 32*3)]

        exists, idx = self.element_exists([detection.nr_batch for detection in self.detections], batch_nr)
        if exists:
            nr_batch, time, nr_persons, x_positions, y_positions, z_positions, _, face_detected = self.detections[idx]
            self.detections[idx] = Persons(nr_batch, time, nr_persons, x_positions, y_positions, z_positions, colour_vectors, face_detected)

        self.compare_hoc(idx) # Temp for Hoc

    def callback_face(self, data: ColourCheckedTarget, amount_detections_stored: int = 100) -> None:
        """ Add the latest Face detection to the storage."""
        batch_nr = data.batch_nr
        face_detections = data.face_detections
        exists, idx = self.element_exists([detection.nr_batch for detection in self.detections], batch_nr)
        if exists:
            nr_batch, time, nr_persons, x_positions, y_positions, z_positions, colour_vectors, _ = self.detections[idx]
            self.detections[idx] = Persons(nr_batch, time, nr_persons, x_positions, y_positions, z_positions,
                                           colour_vectors, face_detections)

    def callback_persons(self, data: DetectedPerson) -> None:
        """ Add the latest detected persons from people_detection to the storage."""
        time = data.time
        nr_batch = data.nr_batch
        nr_persons = data.nr_persons
        x_positions = data.x_positions
        y_positions = data.y_positions
        z_positions = data.z_positions

        if len(self.detections) >= 100:
            self.detections.pop(0)
        colour_vectors = [None] * nr_persons
        face_detected = [None] * nr_persons
        self.detections.append(Persons(nr_batch, time, nr_persons, x_positions, y_positions, z_positions, colour_vectors, face_detected))
        self.new_detections = True
        # rospy.loginfo([person.nr_batch for person in self.detections])
        self.do_data_association(self.detections[-1]) # Temp for DA

    def do_data_association(self, detection: Persons) -> None:
        """ Perform data association based on Euclidean distance between latest and the input. Also updates the
        latest measurement.

        :param detection: [nr_batch, time, nr_persons, x_positions, y_positions, z_positions, colour, face]
        """
        nr_batch, time, nr_persons, x_positions, y_positions, z_positions, _, _ = detection

        if nr_persons <= 0:  # Return if there are no persons in detection
            return

        if len(self.data_data_association) < 1:  # Check if it is the first detection, if so select first target
            self.data_data_association.append([nr_batch, 0])
            rospy.loginfo("new data")
            return

        if nr_batch < self.data_data_association[-1][0]:  # Return if the detection is "old"
            return

        smallest_distance = None
        person = None
        for idx in range(nr_persons):
            idx_last = self.data_data_association[-1][-1]
            exists, idx_batch = self.element_exists([detection.nr_batch for detection in self.detections], self.data_data_association[-1][0])
            if not exists:
                rospy.loginfo("batch does not exist")
                return
            tracked = tuple([self.detections[idx_batch].x_positions[idx_last], self.detections[idx_batch].y_positions[idx_last], self.detections[idx_batch].z_positions[idx_last]])
            distance = self.euclidean_distance(tracked,
                                               tuple([x_positions[idx], y_positions[idx], z_positions[idx]]))
            if smallest_distance is None:
                person = idx
                smallest_distance = distance
            elif distance < smallest_distance:
                person = idx
                smallest_distance = distance

        self.data_data_association.append([nr_batch, person])
        if self.ukf_data_association.current_time < time:
            self.ukf_data_association.update(time, [x_positions[person], y_positions[person], 0])
        rospy.loginfo(self.data_data_association)

    # def redo_data_association(self, start_batch: int, idx_person: int):
    #     """ Redo the data association from a start batch nr.
    #     Assumption:
    #     * The start batch number is in the stored data
    #     """
        # batch_numbers = [detection.nr_batch for detection in self.detections]
        # # rospy.loginfo("batch_nr: %s, start_batch:%s ", batch_numbers, start_batch)
        #
        # exists, idx = self.element_exists(batch_numbers, start_batch)
        #
        # if not exists:
        #     rospy.logerr("Not Possible to redo data association")
        #     return
        #
        # self.ukf_data_association = copy.deepcopy(self.ukf_confirmed)
        #
        # nr_batch, time, nr_persons, x_positions, y_positions, z_positions = self.detections[idx]
        #
        # self.data_data_association = [Detection(nr_batch, time, idx_person, x_positions[idx_person], y_positions[idx_person],
        #                                   z_positions[idx_person])]
        #
        # for detection in self.detections[idx + 1:]:
        #     self.do_data_association(detection)
        # rospy.loginfo("Redone data association")

    def plot_tracker(self): #TODO redo with new data
        """ Plot the trackers on a camera frame and publish it.
        This can be used to visualise all the output from the trackers. [x,y coords] Currently not for depth
        """
        if len(self.data_data_association) >= 1:
            latest_image = self.latest_image
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(latest_image, desired_encoding='passthrough')

            current_time = float(rospy.get_time())
            if self.ukf_data_association.current_time < current_time:  # Get prediction for current time
                ukf_predict = copy.deepcopy(self.ukf_data_association)
                ukf_predict.predict(current_time)
            else:
                ukf_predict = self.ukf_data_association

            x_hoc = int(self.ukf_confirmed.kf.x[0])
            y_hoc = int(self.ukf_confirmed.kf.x[2])

            x_position = int(ukf_predict.kf.x[0])
            y_position = int(ukf_predict.kf.x[2])

            # Blue = HoC ukf latest input measurement
            # Red = UKF prediction location
            # Green = Data association

            cv2.circle(cv_image, (x_hoc, y_hoc), 5, (255, 0, 0, 50), -1)  # plot latest hoc measurement blue
            cv2.circle(cv_image, (x_position, y_position), 5, (0, 0, 255, 50),
                       -1)  # plot ukf prediction measurement red
            # cv2.circle(cv_image, (self.data_data_association[-1][-3], self.data_data_association[-1][-2]), 5, (0, 255, 0, 20),
            #            -1)  # plot latest data ass. measurement green

            tracker_image = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")

            # rospy.loginfo("predict x: %s, y: %s; measured x: %s, y: %s, HoC x: %s, y: %s", x_position, y_position,
            #               self.data_data_association[-1][-3], self.data_data_association[-1][-2], x_hoc, y_hoc)

            self.publisher_debug.publish(tracker_image)

    # def update_ukf_tracker(self, ukf: UKF, ukf_update_data, ukf_data, time):   #TODO update
    #     """ Update the known UKF up until given index."""
    #
    #     if ukf.current_time < time:
    #         for entry in ukf_update_data:
    #             z = [entry.x, entry.y, entry.z]
    #             ukf.update(entry.time, z)
    #
    #         ukf_data.append(ukf_update_data)

    # def reidentify_target(self):    #TODO rewrite with check with all past data
    #     """ Check if the target that is being followed is still the correct one."""
    #     # rospy.loginfo("H: %s, F: %s", self.new_hoc_detection, self.new_face_detection)
    #     if not self.new_face_detection and not self.new_hoc_detection:  # No new re-identification features found
    #         rospy.loginfo("None")
    #         current_time = float(rospy.get_time())
    #
    #         times = [self.time_hoc_detection_sec, self.time_face_detection_sec]
    #         max_index, max_time = max(enumerate(times), key=lambda x: x[1] if x[1] is not None else float('-inf'))  # Find which identifier has been updated last
    #
    #         if max_time is None:   # In case there are no detections
    #             return
    #
    #         if current_time - max_time > 5:
    #             values = [self.hoc_detections[-1], self.face_detections[-1]]
    #             if values[max_index].x > CAMERA_WIDTH - 50 or values[max_index].x < 50:
    #                 rospy.loginfo("Target Lost: Out Of Frame")
    #             else:
    #                 rospy.loginfo("Target Lost: No re-identifier found")
    #         return
    #
    #
    #     if self.new_face_detection and self.new_hoc_detection:  # both a face and HoC update
    #         rospy.loginfo("Face and HoC")
    #         self.new_face_detection = False
    #         self.new_hoc_detection = False
    #         return
    #
    #
    #
    #     if self.new_face_detection:  # Only face update
    #         face_detection = self.face_detections[-1]
    #
    #         try:    # If there is a hoc_detection work with this
    #             hoc_detection = self.hoc_detections[-1]
    #         except IndexError:
    #             hoc_detection = Detection(0, 0, 0, 0, 0, 0)
    #             rospy.loginfo("Only Face - IndexError")
    #
    #         if face_detection.time < hoc_detection.time:    # If HoC is further along than face measurements
    #             rospy.loginfo("Only Face - face time < hoc time")
    #         else:   # Face newer than recent HoC measurement
    #             rospy.loginfo("Only Face - face time >= hoc time")
    #
    #             # Check DA for face
    #             exists, idx = self.element_exists(self.data_data_association, face_detection)
    #             if exists:  # Update the face UKF
    #                 self.update_ukf_tracker(self.ukf_face, self.data_data_association[:idx+1], self.data_face, face_detection.time)
    #             else:   # Redo the data association
    #                 self.redo_data_association(face_detection.batch_nr, face_detection.idx_person)
    #
    #         self.new_face_detection = False
    #         return
    #
    #
    #     if self.new_hoc_detection:  # Only Hoc Update
    #         hoc_detection = self.hoc_detections[-1]
    #
    #         try:    # If there is a hoc_detection work with this
    #             face_detection = self.face_detections[-1]
    #         except IndexError:
    #             face_detection = Detection(0, 0, 0, 0, 0, 0)
    #             rospy.loginfo("Only Hoc - IndexError")
    #
    #         if hoc_detection.time < face_detection.time:    # If Face is further along than Hoc measurements
    #             rospy.loginfo("Only HoC - HoC time < face time")
    #         else:   # HoC newer than recent Face measurement
    #             rospy.loginfo("Only HoC - Hoc time >= face time")
    #
    #             # Check DA for HoC
    #             exists, idx = self.element_exists(self.data_data_association, hoc_detection)
    #             if exists:  # Update the HoC UKF
    #                 self.update_ukf_tracker(self.ukf_hoc, self.data_data_association[:idx+1], self.data_hoc, hoc_detection.time)
    #             else:   # Redo the data association
    #                 self.redo_data_association(hoc_detection.batch_nr, hoc_detection.idx_person)
    #
    #         self.new_hoc_detection = False
    #         return
    #
    #     rospy.loginfo("Only HoC")
    #     self.new_hoc_detection = False
    #     return
    #     # if self.ukf_hoc.current_time > self.ukf_confirmed.current_time and self.ukf_face.current_time > self.ukf_confirmed.current_time:    # if both trackers further than confirmed tracker
    #         # update confirmed tracker with data form the "oldest tracker"
    #         # -> How to make sure that both UKFs are consistent?

    def compare_hoc(self, idx_detection):
        """ Compare newly detected persons to previously detected target."""
        bridge = CvBridge()
        match = False
        idx_person = None

        person_vectors = [detection.colour_vector for detection in self.detections[idx_detection]]

        if len(self.HoC_detections) < 1:
            self.HoC_detections.append(person_vectors[0])
            idx_person = 0
            match = True
        else:
            flag = False
            for Hoc_detection in self.HoC_detections:
                for idx_person, vector in enumerate(person_vectors):
                    distance = self.euclidean_distance(vector, Hoc_detection)
                    if distance < 0.25:
                        # rospy.loginfo(str(idx_person) + " " + str(distance))
                        if len(self.HoC_detections) < 5:
                            self.HoC_detections.append(vector)
                        else:
                            self.HoC_detections = self.HoC_detections[1:] + [vector]
                        flag = True
                        match = True
                        break
                if flag:
                    break

        if match:
            if len(self.data_hoc) > 10:
                self.data_hoc.pop(0)
            self.data_hoc.append([self.detections[idx_detection].nr_batch, self.detections[idx_detection].x_positions[idx_person], self.detections[idx_detection].y_positions[idx_person], self.detections[idx_detection].z_positions[idx_person]] )
        # return match, idx_person


    def loop(self):
        """ Loop that repeats itself at self.rate. Can be used to execute methods at given rate. """
        while not rospy.is_shutdown():
            # if self.new_detections:  # Do data association with most recent detection.

            # if self.latest_image is not None:
            #     self.plot_tracker()

            # self.reidentify_target()  # run the re-identify function at the given frequency
            # rospy.loginfo([detection.face_detected for detection in self.detections])

            self.rate.sleep()


if __name__ == '__main__':
    try:
        node_pt = PeopleTracker()
        node_pt.loop()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
