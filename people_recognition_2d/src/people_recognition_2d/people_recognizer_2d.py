# Python modules
import copy
import time

import math
# ROS modules
import rospy
from cv_bridge import CvBridge
from image_recognition_msgs.msg import Recognition, FaceProperties
from image_recognition_msgs.srv import Recognize, GetFaceProperties
# Image recognition repository modules
from image_recognition_util import image_writer
# People recognition repository modules
from people_recognition_msgs.msg import Person2D
from rospy import ServiceException
from sensor_msgs.msg import Image, RegionOfInterest


def _get_and_wait_for_service(srv_name, srv_class):
    """
    Function to start and wait for dependent service

    :param: srv_name: Service name
    :param: srv_class: Service class
    :return: started ServiceProxy object
    """
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
    try:
        response = srv(args)
    except Exception as e:
        rospy.logwarn("{} service call failed: {}".format(srv.resolved_name, e))
        raise
    else:
        return response


class PeopleRecognizer2D(object):
    def __init__(self, pose_estimation_srv_name, face_recognition_srv_name,
                 face_properties_srv_name, color_extractor_srv_name,
                 enable_age_gender_detection, enable_shirt_color_extraction):

        self._pose_estimation_srv = _get_and_wait_for_service(pose_estimation_srv_name, Recognize)
        self._face_recognition_srv = _get_and_wait_for_service(face_recognition_srv_name, Recognize)

        self._enable_age_gender_detection = enable_age_gender_detection
        self._enable_shirt_color_extraction = enable_shirt_color_extraction

        if self._enable_age_gender_detection:
            self._face_properties_srv = _get_and_wait_for_service(face_properties_srv_name, GetFaceProperties)

        if self._enable_shirt_color_extraction:
            self._color_extractor_srv = _get_and_wait_for_service(color_extractor_srv_name, Recognize)

        self._bridge = CvBridge()

        rospy.loginfo("People recognizer 2D initialized")

    @staticmethod
    def _get_recognitions_with_label(label, recognitions):
        def _is_label_recognition(recognition):
            for p in recognition.categorical_distribution.probabilities:
                if p.label == label and p.probability > recognition.categorical_distribution.unknown_probability:
                    return True
            return False

        return [r for r in recognitions if _is_label_recognition(r)]

    @staticmethod
    def _get_face_rois_ids_pose_estimation(recognitions):
        """
        Get ROIs of faces from pose estimation using the nose, left ear
        and right ear

        :param: recognitions from pose estimation
        """
        nose_recognitions = PeopleRecognizer2D._get_recognitions_with_label("Nose", recognitions)
        left_ear_recognitions = PeopleRecognizer2D._get_recognitions_with_label("LEar", recognitions)
        right_ear_recognitions = PeopleRecognizer2D._get_recognitions_with_label("REar", recognitions)

        rois = []
        group_ids = []
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
    def _get_body_parts_pose_estimation(group_id, recognitions):
        """
        Get a list of all bodyparts associated with a particular group ID

        :param: group_id: The group ID of the bodyparts to be fetched
        :param: recognitions: All bodyparts received from pose estimation
        :return: List of body_parts
        """
        return [r for r in recognitions if r.group_id == group_id]

    @staticmethod
    def _get_container_recognition(roi, recognitions, padding_factor=0.1):
        """
        Associate pose estimation ROI with best pose estimation face ROI

        :param: roi: pose estimation face roi
        :param: recognitions: face recognitions
        """
        x = roi.x_offset + .5 * roi.width
        y = roi.y_offset + .5 * roi.height

        def _point_in_roi(x, y, roi):
            return roi.x_offset <= x <= roi.x_offset + roi.width and roi.y_offset <= y <= roi.y_offset + roi.height

        best = None
        if recognitions:
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
        # ROI needs to be at least 1 pixel in size
        return image[roi.y_offset:roi.y_offset + max(roi.height, 1),
               roi.x_offset:roi.x_offset + max(roi.width, 1)]

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
    def _object_colors_to_label(object_colors, object_name):
        """
        Convert object colors array to label string

        :param: object_colors: Array to colors
        :return: string label
        """
        label = f" {object_name}  colors:"
        for color in object_colors:
            label += " {}".format(color)
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

    @staticmethod
    def _hair_roi_from_face_roi(face_roi, image_shape):
        """
        Given a ROI for a face, shift the ROI to the person's hair. Assuming the person is upright :/

        :param face_roi: RegionOfInterest
        :param image_shape: tuple of the image shape
        :return: RegionOfInterest
        """
        hair_roi = copy.deepcopy(face_roi)
        hair_roi.height = max(int(face_roi.height/6), 5)
        hair_roi.width = max(face_roi.width, 5)
        hair_roi.y_offset = min(hair_roi.y_offset, image_shape[0] - hair_roi.height)
        rospy.logdebug(f"{face_roi=}, {hair_roi=}, {image_shape=}")
        return hair_roi

    def recognize(self, image_msg):
        assert isinstance(image_msg, Image)
        image = self._bridge.imgmsg_to_cv2(image_msg)

        image_annotations = []
        people = []

        # Pose estimation and face recognition service calls
        rospy.loginfo("Starting pose and face recognition...")
        start_recognize = time.time()
        pose_estimation_response = _get_service_response(self._pose_estimation_srv, image_msg)
        face_recognition_response = _get_service_response(self._face_recognition_srv, image_msg)
        rospy.logdebug("Recognize took %.4f seconds", time.time() - start_recognize)
        rospy.loginfo("_get_face_rois_ids_pose_estimation...")

        # Extract face ROIs and their corresponding group ids from recognitions of pose estimation
        pose_estimation_face_rois, pose_estimation_face_group_ids = PeopleRecognizer2D._get_face_rois_ids_pose_estimation(
            pose_estimation_response.recognitions
        )

        body_parts_array = [
            PeopleRecognizer2D._get_body_parts_pose_estimation(group_id, pose_estimation_response.recognitions)
            for group_id in pose_estimation_face_group_ids
        ]

        face_recognitions = [
            PeopleRecognizer2D._get_container_recognition(pose_estimation_face_roi, face_recognition_response.recognitions)
            for pose_estimation_face_roi in pose_estimation_face_rois
        ]

        face_labels = [PeopleRecognizer2D._get_best_label(r) for r in face_recognitions]

        # Face properties service call
        if self._enable_age_gender_detection:
            rospy.loginfo("_get_face_properties...")
            face_image_msg_array = [self._bridge.cv2_to_imgmsg(PeopleRecognizer2D._image_from_roi(image, r.roi), "bgr8") for
                                    r in face_recognitions]
            face_properties_response = _get_service_response(self._face_properties_srv, face_image_msg_array)
            face_properties_array = face_properties_response.properties_array
        else:
            face_properties_array = [FaceProperties()] * len(face_recognitions)

        # Color Extractor service call
        if self._enable_shirt_color_extraction:
            rospy.loginfo("_get_color_extractor...")
            shirt_colors_array = []
            hair_color_array = []
            for r in face_recognitions:
                shirt_roi = PeopleRecognizer2D._shirt_roi_from_face_roi(r.roi, image.shape)
                hair_roi = PeopleRecognizer2D._hair_roi_from_face_roi(r.roi, image.shape)

                shirt_image_msg = self._bridge.cv2_to_imgmsg(PeopleRecognizer2D._image_from_roi(image, shirt_roi))
                hair_image_msg = self._bridge.cv2_to_imgmsg(PeopleRecognizer2D._image_from_roi(image, hair_roi))
                try:
                    color_shirt_extractor_response = _get_service_response(self._color_extractor_srv, shirt_image_msg)
                    color_hair_extractor_response = _get_service_response(self._color_extractor_srv, hair_image_msg)
                except ServiceException as e:
                    rospy.logerr("Color extractor service request failed: {}".format(e))
                    shirt_colors = []
                    hair_colors = []
                else:
                    shirt_colors = [p.label for p in
                                    color_shirt_extractor_response.recognitions[0].categorical_distribution.probabilities]
                    hair_colors = [p.label for p in
                                    color_hair_extractor_response.recognitions[0].categorical_distribution.probabilities]
                    hair_color = hair_colors[0] if hair_colors else ""

                shirt_colors_array.append(shirt_colors)
                hair_color_array.append(hair_color)
        else:
            shirt_colors_array = [[]] * len(face_recognitions)
            hair_color_array = [""] * len(face_recognitions)

        # Prepare image annotation labels and People message
        for face_label, face_properties, shirt_colors, hair_color, body_parts, face_recognition in zip(face_labels,
                                                                                                        face_properties_array,
                                                                                                        shirt_colors_array,
                                                                                                        hair_color_array,
                                                                                                        body_parts_array,
                                                                                                        face_recognitions):

            temp_label = PeopleRecognizer2D._face_properties_to_label(face_properties) + \
                         PeopleRecognizer2D._object_colors_to_label(shirt_colors, "shirt") + \
                         f" hair color: {hair_color}"

            if face_label:
                image_annotations.append(face_label + " " + temp_label)
            else:
                image_annotations.append(temp_label)

            people.append(Person2D(name=face_label,
                                   age=face_properties.age,
                                   gender=face_properties.gender,
                                   gender_confidence=face_properties.gender_confidence,
                                   shirt_colors=shirt_colors,
                                   hair_color=hair_color,
                                   body_parts=body_parts,
                                   face=face_recognition))

        cv_image = image_writer.get_annotated_cv_image(image,
                                                       recognitions=face_recognitions,
                                                       labels=image_annotations)

        return people, cv_image
