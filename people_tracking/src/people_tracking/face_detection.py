#!/usr/bin/env python
from typing import Tuple, Any
from typing import List, Optional

import rospy
from cv_bridge import CvBridge
import face_recognition

from people_tracking.msg import DetectedPerson, FaceTarget
from std_srvs.srv import Empty, EmptyResponse

NODE_NAME = 'Face'
TOPIC_PREFIX = '/hero/'


class FacialRecognition:
    def __init__(self) -> None:
        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber = rospy.Subscriber(TOPIC_PREFIX + 'person_detections', DetectedPerson, self.callback,
                                           queue_size=1)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'face_detections', FaceTarget, queue_size=2)
        self.reset_service = rospy.Service(TOPIC_PREFIX + NODE_NAME + '/reset', Empty, self.reset)

        # Variables
        self.last_batch_processed = 0
        self.known_face_encodings = []
        self.latest_data = None
        self.encoded_batch = None
        self.encoded = False  # True if encoding process of face was succesful

    def reset(self, request):
        """ Reset all stored variables in Class to their default values."""
        self.last_batch_processed = 0
        self.known_face_encodings = []
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

    def recognize_faces(self, input_images: List, ) -> List[Optional[bool]]:
        """ Check if a face on the given image matches the encoded face(s).
        Assumptions:
        * A face has already been encoded and stored.

        :param input_images: list with images to compare to the stored face embedding vector.
        :return: a list with True if face is the stored face, False if face is not the stored face and
                 None if no face detected.
        """
        bridge = CvBridge()
        match_check = []
        for idx, image in enumerate(input_images):
            image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
            face_location = face_recognition.face_locations(image)

            if len(face_location) <= 0:     # No face detected
                match_check.append(None)
            face_encoding = face_recognition.face_encodings(image, face_location)

            for encoding in face_encoding:
                matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.5)
                if any(matches):
                    match_check.append(True)
                else:
                    match_check.append(False)
        return match_check

    def process_latest_data(self) -> None:
        """ Process the most recent data to see if the correct face is detected. This data will be published to the
            publisher.
        """
        if self.latest_data is None:
            return

        if not self.encoded:
            return

        data = self.latest_data
        time = data.time
        nr_batch = data.nr_batch
        nr_persons = data.nr_persons
        detected_persons = data.detected_persons
        x_positions = data.x_positions
        y_positions = data.y_positions
        z_positions = data.z_positions

        if nr_batch > self.last_batch_processed:
            face_detections = self.recognize_faces(detected_persons)

            msg = FaceTarget()
            msg.time = time
            msg.batch_nr = int(nr_batch)
            msg.nr_persons = nr_persons
            msg.x_positions = x_positions
            msg.y_positions = y_positions
            msg.z_positions = z_positions
            msg.face_detections_valid = [False if value is None else True for value in face_detections]
            msg.face_detections = [value if value is not None else False for value in face_detections]

            self.publisher.publish(msg)
            self.last_batch_processed = nr_batch

        self.latest_data = None  # Clear the latest data after processing
        return

    def callback(self, data: DetectedPerson) -> None:
        """ Store the latest person detections. Encodes the first person in detected_persons as the comparison face.

        :param data: DetectedPerson message.
        """
        self.latest_data = data

        if not self.encoded:  # Define first image with face as comparison face
            if len(data.detected_persons) <= 0:
                return
            rospy.loginfo("Encoding %s", data.nr_batch)
            self.encode_known_faces(data.detected_persons[0])

            if len(self.known_face_encodings) >= 1:
                self.encoded = True
                self.encoded_batch = data.nr_batch
                rospy.loginfo("Encoded Face: %s", self.encoded_batch)

    def main_loop(self) -> None:
        """ Loop to process people detections. """
        while not rospy.is_shutdown():
            self.process_latest_data()
            rospy.sleep(0.001)


if __name__ == '__main__':
    try:
        node_face = FacialRecognition()
        node_face.main_loop()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Failed to launch Facial Recognition Node")
        pass
