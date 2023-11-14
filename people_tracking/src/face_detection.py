#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

# MSGS
from sensor_msgs.msg import Image
from people_tracking.msg import ColourCheckedTarget
from people_tracking.msg import DetectedPerson

NODE_NAME = 'Face'
TOPIC_PREFIX = '/hero/'


import face_recognition


class FaceDetection:
    def __init__(self) -> None:
        # ROS Initialize
        rospy.init_node(NODE_NAME, anonymous=True)
        self.subscriber = rospy.Subscriber(TOPIC_PREFIX + 'person_detections', DetectedPerson, self.callback,
                                           queue_size=1)
        self.publisher = rospy.Publisher(TOPIC_PREFIX + 'face_detections', ColourCheckedTarget, queue_size=2)
        self.publisher_debug = rospy.Publisher(TOPIC_PREFIX + 'debug/Face_debug', Image, queue_size=10)

        # Variables
        self.HoC_detections = []
        self.last_batch_processed = 0
        self.known_face_encodings = []
        self.names = ["target"]

    def encode_known_faces(self, image, model: str = "hog") -> None:
        """
        Loads images in the training directory and builds a dictionary of their
        names and encodings. hog (cpu) cnn (gpu)
        """
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            self.known_face_encodings.append(encoding)
            rospy.loginfo("encoded")

    # def _display_face(self,img, bounding_box):
    #     """
    #     Draws bounding boxes around faces, a caption area, and text captions.
    #     """
    #     top, right, bottom, left = bounding_box
    #     cv2.rectangle(img, (top, left), (bottom, right), (255, 0, 0), 2)


    def recognize_faces(self, image):

        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

        face_locations = face_recognition.face_locations(image)
        rospy.loginfo(face_locations)

        face_encodings = face_recognition.face_encodings(image, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.names[best_match_index]

            face_names.append(name)

            rospy.loginfo(name)


    def callback(self, data):
        time = data.time
        nr_batch = data.nr_batch
        nr_persons = data.nr_persons
        detected_persons = data.detected_persons
        x_positions = data.x_positions
        y_positions = data.y_positions
        z_positions = data.z_positions

        if len(self.known_face_encodings) < 1:
            self.encode_known_faces(detected_persons[0])
        else:
            self.recognize_faces(detected_persons[0])
        #
        # match = False
        # idx_match = None
        #
        # if nr_batch > self.last_batch_processed:
        #     match, idx_match = self.compare_hoc(detected_persons)
        #     if match:
        #         msg = ColourCheckedTarget()
        #         msg.time = time
        #         msg.batch_nr = int(nr_batch)
        #         msg.idx_person = int(idx_match)
        #         msg.x_position = x_positions[idx_match]
        #         msg.y_position = y_positions[idx_match]
        #         msg.z_position = 0  # z_positions[idx_match]
        #
        #         self.publisher.publish(msg)
        #     self.last_batch_processed = nr_batch
        #
        # if nr_persons > 0 and match:
        #     self.publisher_debug.publish(detected_persons[idx_match])


if __name__ == '__main__':
    try:
        node_face = FaceDetection()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        pass
