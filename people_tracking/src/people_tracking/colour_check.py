#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from typing import List

from people_tracking.msg import ColourTarget, DetectedPerson
from std_srvs.srv import Empty, EmptyResponse

NODE_NAME = 'HoC'
TOPIC_PREFIX = '/hero/'


class HOC:
    """Class for the histogram of colour node."""
    def __init__(self) -> None:
        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber = rospy.Subscriber(TOPIC_PREFIX + 'person_detections', DetectedPerson, self.callback,
                                           queue_size=1)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'HoC', ColourTarget, queue_size=2)
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
    def get_vector(image, bins: int = 32) -> List[float]:
        """ Return HSV-colour histogram vector from image.

        :param image: cv2 image to turn into vector.
        :param bins: amount of bins in histogram.
        :return: HSV-colour histogram vector from image.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # Convert image to HSV

        histograms = [cv2.calcHist([hsv_image], [i], None, [bins], [1, 256])
                      for i in range(3)]  # Get color histograms

        histograms = [hist / hist.sum() for hist in histograms]  # Normalize histograms

        vector = np.concatenate(histograms, axis=0).reshape(-1)  # Create colour histogram vector
        return vector.tolist()

    def callback(self, data: DetectedPerson) -> None:
        """ Get the colour vectors for each detected person and publish this."""
        time = data.time
        nr_batch = data.nr_batch
        nr_persons = data.nr_persons
        detected_persons = data.detected_persons
        x_positions = data.x_positions
        y_positions = data.y_positions
        z_positions = data.z_positions

        if nr_batch <= self.last_batch_processed:
            return
        if nr_persons < 1:
            return

        bridge = CvBridge()
        colour_vectors = [self.get_vector(bridge.imgmsg_to_cv2(person, desired_encoding='passthrough')) for person in
                          detected_persons]

        msg = ColourTarget()
        msg.time = time
        msg.nr_batch = nr_batch
        msg.nr_persons = nr_persons
        msg.x_positions = x_positions
        msg.y_positions = y_positions
        msg.z_positions = z_positions
        msg.colour_vectors = [item for sublist in colour_vectors for item in sublist]

        self.publisher.publish(msg)
        self.last_batch_processed = nr_batch


if __name__ == '__main__':
    try:
        node_hoc = HOC()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Failed to launch HoC Node")
        pass
