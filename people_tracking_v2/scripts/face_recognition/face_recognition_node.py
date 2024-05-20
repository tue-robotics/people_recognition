#!/usr/bin/env python
import math
import os
import sys
import diagnostic_updater
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import RegionOfInterest, Image
from std_srvs.srv import Empty
from image_recognition_msgs.msg import (
    Recognition,
    Recognitions,
    CategoryProbability,
    CategoricalDistribution,
)
from image_recognition_msgs.srv import Recognize, Annotate
from image_recognition_util import image_writer
from src.face_recognition.face_recognizer import FaceRecognizer


class OpenfaceROS:
    def __init__(
        self,
        save_images_folder,
        topic_save_images,
        service_save_images,
        topic_publish_result_image,
        service_publish_result_image,
    ):
        """
        OpenfaceROS class that wraps the FaceRecognizer in a ROS node

        :param save_images_folder: path where to store the images
        :param topic_save_images: whether to save images originated from image topic callback
        :param service_save_images: whether to save images originated from a service call
        :param topic_publish_result_image: whether to publish images originated from image topic callback
        :param service_publish_result_image: whether to publish images originated from a serice call
        """

        # Openface ROS
        self._face_recognizer = FaceRecognizer()
        self._save_images_folder = save_images_folder
        self._topic_save_images = topic_save_images
        self._service_save_images = service_save_images
        self._topic_publish_result_image = topic_publish_result_image
        self._service_publish_result_image = service_publish_result_image

        self._bridge = CvBridge()
        self._annotate_srv = rospy.Service("annotate", Annotate, self._annotate_srv)
        self._recognize_srv = rospy.Service("recognize", Recognize, self._recognize_srv)
        self._image_subscriber = rospy.Subscriber("image", Image, self._image_callback)
        self._recognitions_publisher = rospy.Publisher("recognitions", Recognitions, queue_size=10)

        if not self._save_images_folder and (self._topic_save_images or self._service_save_images):
            rospy.logerr("~save_images_folder is not defined but we would like to save images ...")
            rospy.signal_shutdown("")

        if self._topic_publish_result_image or self._service_publish_result_image:
            self._result_image_publisher = rospy.Publisher("result_image", Image, queue_size=10)

        rospy.loginfo("OpenfaceROS initialized:")
        rospy.loginfo(" - save_images_folder=%s", save_images_folder)
        rospy.loginfo(" - topic_save_images=%s", topic_save_images)
        rospy.loginfo(" - service_save_images=%s", service_save_images)
        rospy.loginfo(" - topic_publish_result_image=%s", topic_publish_result_image)
        rospy.loginfo(" - service_publish_result_image=%s", service_publish_result_image)

    def _annotate_srv(self, req):
        # Convert to opencv image
        """
        Annotate service callback that trains the face of the user

        :param req: Face annotation request
        :return: Empty
        """
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            raise Exception("Could not convert to opencv image: %s" % str(e))

        for annotation in req.annotations:
            roi_image = bgr_image[
                annotation.roi.y_offset: annotation.roi.y_offset + annotation.roi.height,
                annotation.roi.x_offset: annotation.roi.x_offset + annotation.roi.width,
            ]

            if self._save_images_folder:
                image_writer.write_annotated(self._save_images_folder, roi_image, annotation.label, True)

            try:
                self._face_recognizer.train(roi_image, annotation.label)
            except Exception as e:
                raise Exception("Could not get representation of face image: %s" % str(e))

            rospy.loginfo("Succesfully learned face of '%s'" % annotation.label)

        return {}

    def _get_recognitions(self, image_msg, save_images, publish_images):
        # Convert to opencv image
        """
        Recognize service callback

        :param req: The input image
        :return: Recognitions
        """
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            raise Exception("Could not convert to opencv image: %s" % str(e))

        # Write raw image
        if save_images:
            image_writer.write_raw(self._save_images_folder, bgr_image)

        images = []
        labels = ['Miguel']
        
        # Call facebet neural network in two stages
        face_recognitions = self._face_recognizer.face_detection(bgr_image)
        distance, labels_pred = self._face_recognizer.detection_recognition(bgr_image, labels, train=True)
        
        # Fill recognitions
        recognitions = []

        # rospy.loginfo("Face recognitions: %s", face_recognitions)

        label_idx = 0
        for fr in face_recognitions:            
            face_recognition = [math.floor(xi) for xi in fr]
            if save_images:
                label = labels_pred[label_idx]
                roi_image = bgr_image[
                    face_recognition[2]: face_recognition[3],
                    face_recognition[0]: face_recognition[1],
                ]
                image_writer.write_annotated(self._save_images_folder, roi_image, label, False)
                images.append(roi_image)
                labels.append(label)
            label = labels_pred[label_idx]
            distance_fr = distance[label_idx]
            recognitions.append(
                Recognition(
                    categorical_distribution=CategoricalDistribution(
                        unknown_probability=0.0,  # TODO: When is it unknown?
                        probabilities=[
                            # This line needs some changing 
                            CategoryProbability(label=label, probability=1.0 / (distance_fr + 0.001)) for l2 in face_recognition
                        ],
                    ),
                    roi=RegionOfInterest(
                        x_offset=face_recognition[0],
                        y_offset=face_recognition[1],
                        width=face_recognition[2] - face_recognition[0],
                        height=face_recognition[3] - face_recognition[1],
                    ),
                )
            )
            label_idx = label_idx + 1

        annotated_original_image = image_writer.get_annotated_cv_image(bgr_image, recognitions)
        if save_images:
            image_writer.write_estimations(
                self._save_images_folder,
                images,
                labels,
                annotated_original_image,
                suffix="_face_recognition",
            )

        if publish_images:
            self._result_image_publisher.publish(self._bridge.cv2_to_imgmsg(annotated_original_image, "bgr8"))

        # Service response
        return recognitions

    def _image_callback(self, image_msg):
        # Comment this exception for beeter debbuging
        try:
            recognitions = self._get_recognitions(
                image_msg,
                save_images=self._topic_save_images,
                publish_images=self._topic_publish_result_image,
            )
        except Exception as e:
            rospy.logerr(str(e))
            return

        self._recognitions_publisher.publish(Recognitions(header=image_msg.header, recognitions=recognitions))

    def _recognize_srv(self, req):
        recognitions = self._get_recognitions(
            req.image,
            save_images=self._service_save_images,
            publish_images=self._service_publish_result_image,
        )

        # Service response
        return {"recognitions": recognitions}


if __name__ == "__main__":
    rospy.init_node("face_recognition")
    try:
        save_images = rospy.get_param("~save_images", True)
        topic_save_images = rospy.get_param("~topic_save_images", False)
        service_save_images = rospy.get_param("~service_save_images", True)
        topic_publish_result_image = rospy.get_param("~topic_publish_result_image", True)
        service_publish_result_image = rospy.get_param("~service_publish_result_image", True)

        save_images_folder = None
        if save_images:
            save_images_folder = os.path.expanduser(rospy.get_param("~save_images_folder", "/tmp/facenet_saved_images"))
    except KeyError as e:
        rospy.logerr("Parameter %s not found" % e)
        sys.exit(1)

    image_recognition_openface = OpenfaceROS(
        save_images_folder,
        topic_save_images,
        service_save_images,
        topic_publish_result_image,
        service_publish_result_image,
    )
    updater = diagnostic_updater.Updater()
    updater.setHardwareID("none")
    updater.add(diagnostic_updater.Heartbeat())
    rospy.Timer(rospy.Duration(1), lambda event: updater.force_update())

    rospy.spin()
