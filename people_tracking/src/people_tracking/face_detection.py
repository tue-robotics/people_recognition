#!/usr/bin/env python
from typing import Tuple, Any

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
import face_recognition

# MSGS
from sensor_msgs.msg import Image
from people_tracking.msg import ColourCheckedTarget
from people_tracking.msg import DetectedPerson

from std_srvs.srv import Empty, EmptyResponse

NODE_NAME = 'Face'
TOPIC_PREFIX = '/hero/'


class FaceDetection:
    def __init__(self) -> None:
        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber = rospy.Subscriber(TOPIC_PREFIX + 'person_detections', DetectedPerson, self.callback,
                                           queue_size=1)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'face_detections', ColourCheckedTarget, queue_size=2)
        self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'debug/Face_debug', Image, queue_size=10)
        self.reset_service = rospy.Service(TOPIC_PREFIX + NODE_NAME + '/reset', Empty, self.reset)

        # Variables
        self.HoC_detections = []
        self.last_batch_processed = 0
        self.known_face_encodings = []
        self.names = ["target"]
        self.latest_data = None
        self.encoded_batch = None
        self.encoded = False    #True if encoding process of face was succesful

    def reset(self, request):
        """ Reset all stored variables in Class to their default values."""
        self.HoC_detections = []
        self.last_batch_processed = 0
        self.known_face_encodings = []
        self.names = ["target"]
        self.latest_data = None
        self.encoded_batch = None
        self.encoded = False
        return EmptyResponse()

    def encode_known_faces(self, image, model: str = "hog") -> None:
        """
        From the given image, encode the face and store it.

        :param image: image to find the face to encode on.
        :param model: method to use for encoding. Either hog (cpu) or cnn (gpu).
        """
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            self.known_face_encodings.append(encoding)

    def recognize_faces(self, input_images) -> Tuple[bool, Any]:
        """ Check if a face on the given image matches the encoded face(s).
        Assumptions:
        * A face has already been encoded and stored.
        * Unknown faces get the name Unknown.

        :param input_images: list with images to find a face to check on.
        :return: match, idx_match, match is true if match found,
         and it returns the idx of the element this was found on from the input list.
        """

        bridge = CvBridge()
        for idx, image in enumerate(input_images):
            print(idx)
            image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
            face_location = face_recognition.face_locations(image)
            face_encoding = face_recognition.face_encodings(image, face_location)

            for encoding in face_encoding:
                matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.6)  # True
                # print(face_recognition.face_distance(self.known_face_encodings, encoding))
                # if match, False if no match.
                if any(matches):
                    return True, idx
        return False, None

    def process_latest_data(self):
        """ Process the most recent data to see if the correct face is detected."""

        if self.latest_data is not None and self.encoded:
            data = self.latest_data
            time = data.time
            nr_batch = data.nr_batch
            nr_persons = data.nr_persons

            detected_persons = data.detected_persons  # images
            x_positions = data.x_positions
            y_positions = data.y_positions
            z_positions = data.z_positions

            match = False
            idx_match = None

            if nr_batch > self.last_batch_processed:
                face_detections = [False] * nr_persons
                match, idx_match = self.recognize_faces(detected_persons)
                if match:
                    face_detections[idx_match] = True
                msg = ColourCheckedTarget()
                msg.time = time
                msg.batch_nr = int(nr_batch)
                msg.nr_persons = nr_persons
                msg.x_positions = x_positions
                msg.y_positions = y_positions
                msg.z_positions = z_positions
                msg.face_detections = face_detections

                self.publisher.publish(msg)
                self.last_batch_processed = nr_batch

            if nr_persons > 0 and match:
                self.publisher_debug.publish(detected_persons[idx_match])

            self.latest_data = None  # Clear the latest image after processing
            return
        return

    def callback(self, data):
        self.latest_data = data

        if not self.encoded:  # Define first image with face as target,
                              # make sure that this is done before "throwing" away data
            if len(data.detected_persons) > 0:
                rospy.loginfo("encoding %s", data.nr_batch)
                self.encode_known_faces(data.detected_persons[0])
                if len(self.known_face_encodings) >= 1:
                    self.encoded = True
                    self.encoded_batch = data.nr_batch
                    rospy.loginfo("encoded Face: %s", self.encoded_batch)

    def main_loop(self):
        """ Main loop that makes sure only the latest images are processed. """
        while not rospy.is_shutdown():
            self.process_latest_data()

            rospy.sleep(0.001)


if __name__ == '__main__':
    try:
        node_face = FaceDetection()
        node_face.main_loop()
    except rospy.exceptions.ROSInterruptException:
        pass
