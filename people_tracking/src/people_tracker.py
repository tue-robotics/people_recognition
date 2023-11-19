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
from people_tracking.msg import ColourCheckedTarget
from people_tracking.msg import DetectedPerson

# SrvS
from std_srvs.srv import Empty

NODE_NAME = 'people_tracker'
TOPIC_PREFIX = '/hero/'

laptop = sys.argv[1]
name_subscriber_RGB = 'video_frames' if laptop == "True" else '/hero/head_rgbd_sensor/rgb/image_raw'

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480



class PeopleTracker:
    def __init__(self) -> None:

        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber_HoC = rospy.Subscriber(TOPIC_PREFIX + 'HoC', ColourCheckedTarget,
                                               self.callback_hoc, queue_size=5)
        self.subscriber_Face = rospy.Subscriber(TOPIC_PREFIX + 'face_detections', ColourCheckedTarget,
                                                self.callback_face, queue_size=5)
        self.subscriber_persons = rospy.Subscriber(TOPIC_PREFIX + 'person_detections', DetectedPerson,
                                                   self.callback_persons, queue_size=5)
        self.subscriber_frames = rospy.Subscriber(name_subscriber_RGB, Image, self.get_latest_image, queue_size=1)

        self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'debug/people_tracker', Image, queue_size=10)
        self.rate = rospy.Rate(20)  # 20hz

        # Create a ROS Service Proxy for the color histogram reset service
        rospy.wait_for_service(TOPIC_PREFIX + 'HoC/reset')
        self.hoc_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'HoC/reset', Empty)
        rospy.wait_for_service(TOPIC_PREFIX + 'Face/reset')
        self.face_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'Face/reset', Empty)
        rospy.wait_for_service(TOPIC_PREFIX + 'person_detection/reset')
        self.detection_reset_proxy = rospy.ServiceProxy(TOPIC_PREFIX + 'person_detection/reset', Empty)

        # Variables
        self.hoc_detections = [] # latest detected persons with hoc
        self.face_detections = []   # latest detected persons with face
        self.new_reidentify_hoc = False
        self.time_received_hoc = None
        self.new_reidentify_face = False
        self.time_received_face = None

        self.detections = [] # All persons detected per frame

        self.latest_image = None
        self.tracked_data = []  # Data put into the prediction UKF


        self.ukf_face = UKF()
        self.data_confirmed_face = [] # list of data added to confirmed and not to face

        self.ukf_confirmed = UKF()
        self.ukf_prediction = UKF()

    def reset(self):
        """ Reset all stored variables in Class to their default values."""
        self.latest_image = None
        self.tracked_data = []

        self.ukf_confirmed = UKF()
        self.ukf_prediction = UKF()

    @staticmethod
    def euclidean_distance(point1, point2):
        """ Calculate the Euclidean distance between two points.

        :param point1: A tuple or list representing the coordinates of the first point.
        :param point2: A tuple or list representing the coordinates of the second point.
        :return: The Euclidean distance between the two points.
        """
        if len(point1) != len(point2):
            raise ValueError("Not the same dimensions")

        squared_sum = sum((coord2 - coord1) ** 2 for coord1, coord2 in zip(point1, point2))
        distance = math.sqrt(squared_sum)
        return distance

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

    def callback_hoc(self, data):
        """ Add the latest HoC detection to the storage."""
        time = data.time
        batch_nr = data.batch_nr
        idx_person = data.idx_person
        x_position = data.x_position
        y_position = data.y_position
        z_position = data.z_position

        if len(self.hoc_detections) >= 5:
            self.hoc_detections = self.hoc_detections[1:]
        self.hoc_detections.append([batch_nr, idx_person, time, x_position, y_position, z_position])

        self.new_reidentify_hoc = True
        self.time_received_hoc = float(rospy.get_time())

    def callback_face(self, data):
        """ Add the latest Face detection to the storage."""
        time = data.time
        batch_nr = data.batch_nr
        idx_person = data.idx_person
        x_position = data.x_position
        y_position = data.y_position
        z_position = data.z_position

        if len(self.face_detections) >= 5:
            self.face_detections = self.face_detections[1:]
        self.face_detections.append([batch_nr, idx_person, time, x_position, y_position, z_position])

        self.new_reidentify_face = True
        self.time_received_face = float(rospy.get_time())

    def callback_persons(self, data) -> None:
        """ Update the ukf_prediction using the closest image. (Data Association based on distance)."""
        time = data.time
        nr_batch = data.nr_batch
        nr_persons = data.nr_persons
        x_positions = data.x_positions
        y_positions = data.y_positions
        z_positions = data.z_positions

        self.detections.append([nr_batch, time, nr_persons, x_positions, y_positions, z_positions])
        self.data_association(self.detections[-1]) #this one could case the to far ahead  data in reindentifier?

    def data_association(self, detection:  List[Union[int, float, int, List[int], List[int], List[int]]]):
        """ Perform data association based on euclidean distance between latest and the input. Also updates the
        latest measurement.

        :param detection: [nr_batch, time, nr_persons, x_positions, y_positions, z_positions]
        """
        nr_batch, time, nr_persons, x_positions, y_positions, z_positions = detection

        if len(self.tracked_data) < 1:  # Check if it is the first detection
            self.tracked_data.append(
                [nr_batch, 0, time, x_positions[0], y_positions[0], z_positions[0]])
            return

        if time < self.tracked_data[-1][0]:     # Return if the detection is "old"
            return

        if nr_persons <= 0:  # Return if there are no persons in detection
            return

        smallest_distance = None
        person = None
        for idx in range(nr_persons):
            tracked = tuple(self.tracked_data[-1][-3:])
            distance = self.euclidean_distance(tracked,
                                               tuple([x_positions[idx], y_positions[idx], z_positions[idx]]))
            if smallest_distance is None:
                person = idx
                smallest_distance = distance
            elif distance < smallest_distance:
                person = idx
                smallest_distance = distance

        self.tracked_data.append(
            [nr_batch, person, time, x_positions[person], y_positions[person], z_positions[person]])
        self.ukf_prediction.update(time, [x_positions[person], y_positions[person], 0])  # ToDo move make sure to add time stamp check?

    def redo_data_association(self, start_batch, idx_person):
        """ Redo the data association from a start batch nr.
        Assumption:
        * The start batch number is in the stored data
        """
        batch_numbers = [detection[0] for detection in self.detections]
        rospy.loginfo("batch_nr: %s, start_batch:%s ", batch_numbers, start_batch)

        exists, idx = self.element_exists(batch_numbers, start_batch)

        if not exists:
            rospy.loginfo("Not Possible to redo data association")
            return

        self.ukf_prediction = copy.deepcopy(self.ukf_confirmed)

        nr_batch, time, nr_persons, x_positions, y_positions, z_positions = self.detections[idx]

        self.tracked_data = [[nr_batch, time, idx_person, x_positions[idx_person], y_positions[idx_person], z_positions[idx_person]]]

        for detection in self.detections[idx+1:]:
            self.data_association(detection)
        rospy.loginfo("Redone data association")

    def get_latest_image(self, data):
        """ Get the most recent frame/image from the camera."""
        self.latest_image = data

    def plot_tracker(self):
        """ Plot the trackers on a camera frame and publish it.
        This can be used to visualise all the output from the trackers. [x,y coords] Currently not for depth
        """
        if len(self.tracked_data) >= 1:
            latest_image = self.latest_image
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(latest_image, desired_encoding='passthrough')

            current_time = float(rospy.get_time())
            if self.ukf_prediction.current_time < current_time:
                ukf_predict = copy.deepcopy(self.ukf_prediction)
                ukf_predict.predict(current_time)
            else:
                ukf_predict = self.ukf_prediction

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
            cv2.circle(cv_image, (self.tracked_data[-1][-3], self.tracked_data[-1][-2]), 5, (0, 255, 0, 20),
                       -1)  # plot latest data ass. measurement green

            tracker_image = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")

            # rospy.loginfo("predict x: %s, y: %s; measured x: %s, y: %s, HoC x: %s, y: %s", x_position, y_position,
            #               self.tracked_data[-1][-3], self.tracked_data[-1][-2], x_hoc, y_hoc)

            self.publisher_debug.publish(tracker_image)


    def update_confirmed_tracker(self, update_idx, type):
        """ Update the known UKF up until given index."""

        if type is "hoc":
            self.data_confirmed_face.append(self.tracked_data[:update_idx])

            update_data = self.tracked_data[:update_idx + 1][:]
            for entry in update_data:
                z = [entry[3], entry[4], entry[5]]
                self.ukf_confirmed.update(entry[2], z)
            self.tracked_data = self.tracked_data[update_idx:][:]

        if type is "face":
            rospy.loginfo("face update not done")


        # rospy.loginfo("Confirmed Update")

    def reidentify_target(self):
        """ Check if the target that is being followed is still the correct one."""
        # rospy.loginfo("H: %s, F: %s", self.new_reidentify_hoc, self.new_reidentify_face)
        if not self.new_reidentify_face and not self.new_reidentify_hoc:    # No new re-identification features found
            rospy.loginfo("None")
            current_time = float(rospy.get_time())
            if self.time_received_hoc is None:  # Check that there is a measurement
                return
            if current_time - self.time_received_hoc > 5: # ToDo still determine correct time for this
                if self.tracked_data[-1][-3] > CAMERA_WIDTH - 50 or self.tracked_data[-1][-3] < 50:
                    rospy.loginfo("Target Lost: Out Of Frame")
                else:
                    rospy.loginfo("Target Lost: No re-identifier found for a while")
                    # DA + set new Hoc from this? --> Look into best option
            return

        if self.new_reidentify_face and self.new_reidentify_hoc: # both a face and HoC update
            face_exists, face_idx = self.element_exists(self.tracked_data, self.face_detections[-1])
            hoc_exists, hoc_idx = self.element_exists(self.tracked_data, self.hoc_detections[-1])

            if face_exists and hoc_exists: # both face and hoc in same DA line, so assume still on correct target
               rospy.loginfo("Both")
               # Update UKF and prune data up until newest
               if self.face_detections[-1][0] > self.hoc_detections[-1][0]:
                    self.update_confirmed_tracker(face_idx, "face")
                    if len(self.detections) > 10:
                        self.detections = self.detections[-10:]
               else:
                   self.update_confirmed_tracker(hoc_idx, "hoc")

            else:   # if gone potentially to the wrong path
                self.ukf_prediction = copy.deepcopy(self.ukf_confirmed)
                # Reset tracked_data with the latest detection (either face or hoc)
                if face_exists: # if face in there -> Update UKF and prune till face, reset hoc to face data detection
                    rospy.loginfo("Both - Face")
                    self.update_confirmed_tracker(face_idx, "face")
                    if len(self.detections) > 10:
                        self.detections = self.detections[-10:]
                    self.redo_data_association(self.face_detections[-1][0], self.face_detections[-1][2])
                    #  TODO reset hoc to face detection point
                elif hoc_exists:
                    rospy.loginfo("Both-HoC")
                    # TODO if face not in there but HoC in there -> go to back to last known and check in all data from face measurement
                    # and continue from there with DA and check if hoc reapears (if hoc is newwr) if not, re-set and update hoc from old data from new point with da
            self.new_reidentify_face = False
            self.new_reidentify_hoc = False
            return

        if self.new_reidentify_face:   # Only face update
            rospy.loginfo("Only Face")
            #make sure that this is still in line with hoc ?
            face_exists, face_idx = self.element_exists(self.tracked_data, self.face_detections[-1])
            if face_exists:
                self.update_confirmed_tracker(face_idx, "face")
                if len(self.detections) > 10:
                    self.detections = self.detections[-10:]
            else:
                self.redo_data_association(self.face_detections[-1][0], self.face_detections[-1][1])

            self.new_reidentify_face = False
            return

        if self.new_reidentify_hoc:    # Only Hoc Update
            rospy.loginfo("Only HoC")
            # Make sure that this is still in line with face ?
            hoc_exists, hoc_idx = self.element_exists(self.tracked_data, self.hoc_detections[-1])
            if not hoc_exists:
                if self.hoc_detections[-1][0] >= self.tracked_data[-1][0]:
                    self.new_reidentify_hoc = True
                else:
                    self.redo_data_association(self.hoc_detections[-1][0], self.hoc_detections[-1][1])
            else:
                self.update_confirmed_tracker(hoc_idx)

            self.new_reidentify_hoc = False
            return








    def loop(self):
        """ Loop that repeats itself at self.rate.
            Currently used for publishing the tracker data on an image.
        """
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                self.plot_tracker()

            self.reidentify_target() # run the re-identify function at the given frequency

            # if self.latest_image:
            #     self.reset_color_histogram_node()
            self.rate.sleep()

    def reset_color_histogram_node(self):

        # Call the color histogram reset service
        try:
            response = self.hoc_reset_proxy()
            rospy.loginfo("Color histogram node reset successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to reset color histogram node: %s", str(e))
        try:
            response = self.face_reset_proxy()
            rospy.loginfo("Face node reset successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to reset Face node: %s", str(e))
        try:
            response = self.detection_reset_proxy()
            rospy.loginfo("Detection node reset successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to reset detection node: %s", str(e))


if __name__ == '__main__':
    try:
        node_pt = PeopleTracker()
        node_pt.loop()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
