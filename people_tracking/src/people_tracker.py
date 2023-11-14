#!/usr/bin/env python
import sys
import rospy
import cv2
import math
from cv_bridge import CvBridge
import copy
from UKFclass import *

# MSGS
from sensor_msgs.msg import Image
from people_tracking.msg import ColourCheckedTarget
from people_tracking.msg import DetectedPerson

NODE_NAME = 'people_tracker'
TOPIC_PREFIX = '/hero/'

laptop = sys.argv[1]
name_subscriber_RGB = 'video_frames' if laptop == "True" else '/hero/head_rgbd_sensor/rgb/image_raw'


class PeopleTracker:
    def __init__(self) -> None:

        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber_HoC = rospy.Subscriber(TOPIC_PREFIX + 'HoC', ColourCheckedTarget,
                                               self.callback_HoC, queue_size=1)
        self.subscriber_persons = rospy.Subscriber(TOPIC_PREFIX + 'person_detections', DetectedPerson,
                                                   self.callback_persons, queue_size=1)
        self.subscriber_frames = rospy.Subscriber(name_subscriber_RGB, Image, self.get_latest_image, queue_size=1)

        self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'debug/people_tracker', Image, queue_size=10)
        self.rate = rospy.Rate(20)  # 20hz

        # Variables
        self.latest_image = None
        self.tracked_data = []

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

    def callback_HoC(self, data):
        """ Update the ukf_confirmed with the HoC data, as well as check that the ukf_prediction is still
        following the correct target.
        """
        time = data.time
        batch_nr = data.batch_nr
        idx_person = data.idx_person
        x_position = data.x_position
        y_position = data.y_position
        z_position = data.z_position

        exists, idx = self.element_exists(self.tracked_data,
                                          [batch_nr, idx_person, time, x_position, y_position, z_position])
        # rospy.loginfo("exist: %s, idx: %s", exist, idx)

        if exists:
            update_data = self.tracked_data[:idx + 1][:]
            for entry in update_data:
                z = [entry[3], entry[4], 0]
                self.ukf_confirmed.update(entry[2], z)
            self.tracked_data = self.tracked_data[idx:][:]
        else:  # if gone potentially to wrong path
            self.ukf_prediction = copy.deepcopy(self.ukf_confirmed)
            # TODO add new way to go (redo data association from known point)
            # TODO add somthing that check for long data losses and than resets or stops tracker. e.g. target lost
            # self.tracked_data = [[batch_nr, idx_person, time, x_position, y_position, z_position]]

    def callback_persons(self, data) -> None:
        """ Update the ukf_prediction using the closest image. (Data Association based on distance)."""
        time = data.time
        nr_batch = data.nr_batch
        nr_persons = data.nr_persons
        x_positions = data.x_positions
        y_positions = data.y_positions
        z_positions = data.z_positions

        if len(self.tracked_data) < 1:
            self.tracked_data.append(
                [nr_batch, 0, time, x_positions[0], y_positions[0], z_positions[0]])
            return

        if time > self.tracked_data[-1][0]:
            if nr_persons > 0:
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
                self.ukf_prediction.update(time, [x_positions[person], y_positions[person], 0])

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
                self.ukf_prediction.predict(current_time)

            x_hoc = int(self.ukf_confirmed.kf.x[0])
            y_hoc = int(self.ukf_confirmed.kf.x[2])

            x_position = int(self.ukf_prediction.kf.x[0])
            y_position = int(self.ukf_prediction.kf.x[2])

            # Blue = HoC ukf latest input measurement
            # Red = UKF prediction location
            # Green = Data association

            cv2.circle(cv_image, (x_hoc, y_hoc), 5, (255, 0, 0, 50), -1)  # plot latest hoc measurement blue
            cv2.circle(cv_image, (x_position, y_position), 5, (0, 0, 255, 50), -1)  # plot ukf prediction measurement red
            cv2.circle(cv_image, (self.tracked_data[-1][-3], self.tracked_data[-1][-2]), 5, (0, 255, 0, 20),
                       -1)  # plot latest data ass. measurement green

            tracker_image = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")

            # rospy.loginfo("predict x: %s, y: %s; measured x: %s, y: %s, HoC x: %s, y: %s", x_position, y_position,
            #               self.tracked_data[-1][-3], self.tracked_data[-1][-2], x_hoc, y_hoc)

            self.publisher_debug.publish(tracker_image)

    def loop(self):
        """ Loop that repeats itself at self.rate.
            Currently used for publishing the tracker data on an image.
        """
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                self.plot_tracker()

            self.rate.sleep()


if __name__ == '__main__':
    try:
        node_pt = PeopleTracker()
        node_pt.loop()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
