# Python modules
import time
from contextlib import closing
from multiprocessing import Pool
import cv2
import math
import copy

# ROS modules
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, RegionOfInterest

# Image recognition repository modules
from image_recognition_util import image_writer
from image_recognition_msgs.msg import Recognition, FaceProperties, Person
from image_recognition_msgs.srv import (Recognize, RecognizeResponse,
                                        GetFaceProperties, GetFacePropertiesResponse,
                                        ExtractColour, ExtractColourResponse)

def _get_and_wait_for_service(srv_name, srv_class):
    service = rospy.ServiceProxy('{}'.format(srv_name), srv_class)
    rospy.loginfo("Waiting for service {} ...".format(service.resolved_name))
    service.wait_for_service()
    return service

def _get_service_response(srv, args):
    """
    Method to get service response with checks
    :param: srv: service
    :param: args: Input arguments of the service request
    :return: response
    """
    response = None
    try:
        response = srv(args)
    except rospy.ServiceException as e:
        rospy.logwarn("{} service call failed: {}".format(srv.resolved_name, e))
        return None
    return response

class PeopleDetector(object):
    def __init__(self, openpose_srv_name, openface_srv_name,
            keras_srv_name, colour_extractor_srv_name):

        self._openpose_srv = _get_and_wait_for_service(openpose_srv_name, Recognize)
        self._openface_srv = _get_and_wait_for_service(openface_srv_name, Recognize)
        self._keras_srv = _get_and_wait_for_service(keras_srv_name, GetFaceProperties)
        self._colour_extractor_srv = _get_and_wait_for_service(colour_extractor_srv_name, ExtractColour)

        self._bridge = CvBridge()

        rospy.loginfo("People detector initialized")

    @staticmethod
    def _get_recognitions_with_label(label, recognitions):
        def _is_label_recognition(recognition):
            for p in recognition.categorical_distribution.probabilities:
                if p.label == label and p.probability > recognition.categorical_distribution.unknown_probability:
                    return True
            return False

        return [r for r in recognitions if _is_label_recognition(r)]

    @staticmethod
    def _get_face_rois_ids_openpose(recognitions):
        """
        Get ROIs of faces from openpose recognitions using the nose, left ear
        and right ear
        :param: recognitions from openpose
        """
        nose_recognitions = PeopleDetector._get_recognitions_with_label("Nose", recognitions)
        left_ear_recognitions = PeopleDetector._get_recognitions_with_label("LEar", recognitions)
        right_ear_recognitions = PeopleDetector._get_recognitions_with_label("REar", recognitions)

        rois = list()
        group_ids = list()
        for nose_recognition in nose_recognitions:
            # We assume a vertical head here
            left_size = 50
            right_size = 50
            try:
                left_ear_recognition = next(
                    r for r in left_ear_recognitions if r.group_id == nose_recognition.group_id)

                left_size = math.hypot(left_ear_recognition.roi.x_offset - nose_recognition.roi.x_offset,
                                       left_ear_recognition.roi.y_offset - nose_recognition.roi.y_offset)
            except StopIteration:
                pass
            try:
                right_ear_recognition = next(
                    r for r in right_ear_recognitions if r.group_id == nose_recognition.group_id)

                right_size = math.hypot(right_ear_recognition.roi.x_offset - nose_recognition.roi.x_offset,
                                        right_ear_recognition.roi.y_offset - nose_recognition.roi.y_offset)
            except StopIteration:
                pass

            size = left_size + right_size
            width = int(size)
            height = int(math.sqrt(2) * size)

            rois.append(RegionOfInterest(
                x_offset=max(0, int(nose_recognition.roi.x_offset - .5 * width)),
                y_offset=max(0, int(nose_recognition.roi.y_offset - .5 * height)),
                width=width,
                height=height
            ))
            group_ids.append(nose_recognition.group_id)

        return rois, group_ids

    @staticmethod
    def _get_body_parts_openpose(group_id, recognitions):
        """
        Get a list of all bodyparts associated with a particular group ID
        :param: group_id: The group ID of the bodyparts to be fetched
        :param: recognitions: All bodyparts recieved from openpose
        :return: List of body_parts
        """
        return [r for r in recognitions if r.group_id == group_id]

    @staticmethod
    def _get_container_recognition(roi, recognitions, padding_factor=0.1):
        """
        Associate OpenPose ROI with best OpenPose face ROI
        :param: roi: openpose face roi
        :recognitions: openface recognitions
        """
        x = roi.x_offset + .5 * roi.width
        y = roi.y_offset + .5 * roi.height

        def _point_in_roi(x, y, roi):
            return roi.x_offset <= x <= roi.x_offset + roi.width and roi.y_offset <= y <= roi.y_offset + roi.height

        best = None
        for r in recognitions:
            if _point_in_roi(x, y, r.roi):
                if best:
                    avg_x = r.roi.x_offset + .5 * r.roi.width
                    avg_y = r.roi.y_offset + .5 * r.roi.height
                    best_avg_x = best.roi.x_offset + .5 * best.roi.width
                    best_avg_y = best.roi.y_offset + .5 * best.roi.height
                    if math.hypot(avg_x - x, avg_y - y) > math.hypot(best_avg_x - x, best_avg_y - y):
                        continue
                best = r
        if not best:
            best = Recognition(roi=roi)

        best.roi.x_offset = int(max(0, best.roi.x_offset - padding_factor * best.roi.width))
        best.roi.y_offset = int(max(0, best.roi.y_offset - padding_factor * best.roi.height))
        best.roi.width = int(best.roi.width + best.roi.width * 2 * padding_factor)
        best.roi.height = int(best.roi.height + best.roi.height * 2 * padding_factor)

        return best

    @staticmethod
    def _get_best_label_from_categorical_distribution(c):
        name_probabilities = [p for p in c.probabilities if p.probability > c.unknown_probability]
        if not name_probabilities:
            return None
        return max(c.probabilities, key=lambda p: p.probability)

    @staticmethod
    def _image_from_roi(image, roi):
        return image[roi.y_offset:roi.y_offset + roi.height, roi.x_offset:roi.x_offset + roi.width]

    @staticmethod
    def _get_best_label(recognition):
        best_p = None
        for p in recognition.categorical_distribution.probabilities:
            if p.probability > recognition.categorical_distribution.unknown_probability:
                if best_p and p.probability < best_p.probability:
                    continue
                best_p = p
        if best_p:
            return best_p.label
        else:
            return None

    @staticmethod
    def _face_properties_to_label(face_properties):
        return "{}={} (age={})".format("MALE" if face_properties.gender == FaceProperties.MALE else "FEMALE",
                                        face_properties.gender_confidence,
                                        face_properties.age)

    @staticmethod
    def _shirt_colours_to_label(shirt_colours):
        """
        Convert shirt colours array to label string
        :param: shirt_colours: Array to colours
        :return: string label
        """
        label = " shirt colours:"
        for colour in shirt_colours:
            label += " {}".format(colour)
        return label

    @staticmethod
    def _shirt_roi_from_face_roi(face_roi, image_shape):
        """
        Given a ROI for a face, shift the ROI to the person's shirt. Assuming the person is upright :/
        :param face_roi: RegionOfInterest
        :param image_shape: tuple of the image shape
        :return: RegionOfInterest
        """
        shirt_roi = copy.deepcopy(face_roi)
        shirt_roi.height = max(face_roi.height, 5)
        shirt_roi.width = max(face_roi.width, 5)
        shirt_roi.y_offset += int(face_roi.height * 1.5)
        shirt_roi.y_offset = min(shirt_roi.y_offset, image_shape[0] - shirt_roi.height)
        rospy.logdebug("face_roi: {}, shirt_roi: {}, img.shape: {}".format(face_roi, shirt_roi, image_shape))
        return shirt_roi

    def recognize(self, image_msg):
        assert isinstance(image_msg, Image)
        image = self._bridge.imgmsg_to_cv2(image_msg)

        image_annotations = list()
        people = list()

        # OpenPose and OpenFace service calls
        rospy.loginfo("Starting pose and face recognition...")
        start_recognize = time.time()
        openpose_response = _get_service_response(self._openpose_srv, image_msg)
        assert isinstance(openpose_response, RecognizeResponse)

        openface_response = _get_service_response(self._openface_srv, image_msg)
        assert isinstance(openface_response, RecognizeResponse)
        rospy.logdebug("Recognize took %.4f seconds", time.time() - start_recognize)
        rospy.loginfo("_get_face_rois_ids_openpose...")

        # Extract face ROIs and their corresponding group ids from recognitions of openpose
        openpose_face_rois, openpose_face_group_ids = PeopleDetector._get_face_rois_ids_openpose(openpose_response.recognitions)

        body_parts_array = [PeopleDetector._get_body_parts_openpose(group_id,
            openpose_response.recognitions) for group_id in openpose_face_group_ids]

        face_recognitions = [PeopleDetector._get_container_recognition(openpose_face_roi,
                                                                       openface_response.recognitions)
                             for openpose_face_roi in openpose_face_rois]

        face_labels = [PeopleDetector._get_best_label(r) for r in face_recognitions]

        # Keras service call
        rospy.loginfo("_get_face_properties...")
        face_image_msg_array = [self._bridge.cv2_to_imgmsg(PeopleDetector._image_from_roi(image, r.roi), "bgr8") for r in face_recognitions]
        keras_response = _get_service_response(self._keras_srv, face_image_msg_array)
        assert isinstance(keras_response, GetFacePropertiesResponse)
        face_properties_array = keras_response.properties_array

        # Colour Extractor service call
        rospy.loginfo("_get_colour_extractor...")
        shirt_colours_array = list()
        for r in face_recognitions:
            shirt_roi = PeopleDetector._shirt_roi_from_face_roi(r.roi, image.shape)
            shirt_image_msg = self._bridge.cv2_to_imgmsg(PeopleDetector._image_from_roi(image, shirt_roi))
            colour_extractor_response = _get_service_response(self._colour_extractor_srv, shirt_image_msg)
            assert isinstance(colour_extractor_response, ExtractColourResponse)
            shirt_colours_array.append(colour_extractor_response.colours)

        # Prepare image annotation labels and People message
        for face_label, face_properties, shirt_colours, body_parts in zip(face_labels,
                face_properties_array, shirt_colours_array, body_parts_array):

            temp_label = PeopleDetector._face_properties_to_label(face_properties) + \
                    PeopleDetector._shirt_colours_to_label(shirt_colours)

            if face_label:
                image_annotations.append(face_label + " " + temp_label)
            else:
                image_annotations.append(temp_label)

            people.append(Person(name=face_label,
                                 age=face_properties.age,
                                 gender=face_properties.gender,
                                 gender_confidence=face_properties.gender_confidence,
                                 shirt_colors=shirt_colours,
                                 body_parts=body_parts))

        cv_image = image_writer.get_annotated_cv_image(image,
                                                       recognitions=face_recognitions,
                                                       labels=image_annotations)


        return people, cv_image
