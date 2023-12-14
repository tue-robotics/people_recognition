#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

# MSGS
from sensor_msgs.msg import Image
from people_tracking.msg import ColourCheckedTarget
from people_tracking.msg import DetectedPerson
from people_tracking.msg import ColourTarget

from rospy.numpy_msg import numpy_msg


NODE_NAME = 'HoC'
TOPIC_PREFIX = '/hero/'

from std_srvs.srv import Empty, EmptyResponse

class HOC:
    def __init__(self) -> None:
        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber = rospy.Subscriber(TOPIC_PREFIX + 'person_detections', DetectedPerson, self.callback,
                                           queue_size=1)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'HoC', ColourTarget, queue_size=2)
        # self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'debug/HoC_debug', Image, queue_size=10)

        self.reset_service = rospy.Service(TOPIC_PREFIX + NODE_NAME + '/reset', Empty, self.reset)

        # Variables
        self.HoC_detections = []
        self.last_batch_processed = 0

    def reset(self, request):
        """ Reset all stored variables in Class to their default values."""
        self.HoC_detections = []
        self.last_batch_processed = 0
        return EmptyResponse()

    @staticmethod
    def get_vector(image, bins=32):
        """ Return HSV-colour histogram vector from image.
        :param image: cv2 image to turn into vector
        :param bins: amount of bins in histogram
        :return: HSV-colour histogram vector from image
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # Convert to HSV

        histograms = [cv2.calcHist([hsv_image], [i], None, [bins], [0, 256])
                      for i in range(3)]  # Get color histograms

        histograms = [hist / hist.sum() for hist in histograms]  # Normalize histograms

        vector = np.concatenate(histograms, axis=0).reshape(-1)  # Create colour histogram vector
        return vector.tolist()

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
        y_positions = data.y_positions
        z_positions = data.z_positions

        match = False
        idx_match = None

        if nr_batch <= self.last_batch_processed:
            return
        if nr_persons < 1:
            return
        bridge = CvBridge()
        # match, idx_match = self.compare_hoc(detected_persons)
        colour_vectors = [self.get_vector(bridge.imgmsg_to_cv2(person, desired_encoding='passthrough')) for person in
                          detected_persons]
        # if match:

        msg = ColourTarget()
        msg.time = time
        msg.nr_batch = nr_batch
        msg.nr_persons = nr_persons
        msg.x_positions = x_positions
        msg.y_positions = y_positions
        msg.z_positions = z_positions
        msg.colour_vectors = [item for sublist in colour_vectors for item in sublist]
        # msg.detected_person = detected_persons[idx_match]

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