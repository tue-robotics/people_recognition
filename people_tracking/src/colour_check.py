#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

# MSGS
from people_tracking.msg import ColourCheckedTarget
from people_tracking.msg import DetectedPerson

from sensor_msgs.msg import Image


NODE_NAME = 'HoC'
TOPIC_PREFIX = '/hero/'

class HOC:
    def __init__(self) -> None:

        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber = rospy.Subscriber(TOPIC_PREFIX + 'person_detections', DetectedPerson, self.callback, queue_size = 1)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'HoC', ColourCheckedTarget, queue_size=2)
        # self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'HOCdebug', Image, queue_size=10)

        # Variables
        self.HoC_detections = []
        self.last_batch_processed = 0

    @staticmethod
    def get_vector(image, bins=32):
        """ Return HSV-colour histogram vector from image.
        :param image: cv2 image to turn into vector
        :param bins: amount of bins in histogram
        :return: HSV-colour histogram vector from image
        """
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Get color histograms
        histograms = [cv2.calcHist([hsv_image], [i], None, [bins], [0, 256]) for i in range(3)]

        # Normalize histograms
        histograms = [hist / hist.sum() for hist in histograms]

        # Create Vector
        vector = np.concatenate(histograms, axis=0).reshape(-1)
        return vector

    @staticmethod
    def euclidean(a, b):
        """ Euclidean distance between two vectors. Closer to 0 means better match."""
        return np.linalg.norm(a - b)

    @staticmethod
    def cosine(a, b):
        """ Cosine distance between two vectors. Closer to 1 means a better match."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    def compare_hoc(self, detected_persons):
        """ Compare newly detected persons to previously detected target."""
        bridge = CvBridge()
        match = False
        idx_person = None

        person_vectors = [self.get_vector(bridge.imgmsg_to_cv2(person, desired_encoding='passthrough')) for person in
                          detected_persons]

        if len(self.HoC_detections) < 1:
            self.HoC_detections.append(person_vectors[0])
            idx_person = 0
            match = True
        else:
            flag = False
            for Hoc_detection in self.HoC_detections:
                for idx_person, vector in enumerate(person_vectors):
                    distance = self.euclidean(vector, Hoc_detection)
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
        return match, idx_person

    def callback(self, data):
        time = data.time
        nr_batch = data.nr_batch
        nr_persons = data.nr_persons
        detected_persons = data.detected_persons
        x_positions = data.x_positions

        match = False
        idx_match = None

        if nr_batch > self.last_batch_processed:
            match, idx_match = self.compare_hoc(detected_persons)
            if match:
                msg = ColourCheckedTarget()
                msg.time = time
                msg.batch_nr = int(nr_batch)
                msg.idx_person = int(idx_match)
                msg.x_position = x_positions[idx_match]
                msg.y_position = 0
                msg.z_position = 0

                self.publisher.publish(msg)
            self.last_batch_processed = nr_batch

        # if nr_persons > 0 and match:
        #     self.publisher_debug.publish(detected_persons[idx_match])


if __name__ == '__main__':
    try:
        node_hoc = HOC()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
