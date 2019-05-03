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
from std_msgs.msg import String
from sensor_msgs.msg import RegionOfInterest

# Image recognition repository modules
from image_recognition_msgs.msg import Recognition, FaceProperties, Person
from image_recognition_msgs.srv import Recognize, GetFaceProperties, ExtractColour
from image_recognition_util import image_writer


def _threaded_srv(args):
    """
    Required for calling service in parallel
    """
    srv, kwarg_dict = args
    result = srv(**kwarg_dict)
    del args
    return result

def _get_and_wait_for_services(service_names, service_class, suffix=""):
    services = {s: rospy.ServiceProxy('{}{}'.format(s, suffix), service_class) for s in service_names}
    for service in services.values():
        rospy.loginfo("Waiting for service {} ...".format(service.resolved_name))
        service.wait_for_service()
    return services

class PeopleDetector(object):
    def __init__(self):
        self._recognize_services = _get_and_wait_for_services([
            'openpose',
            'openface'
        ], Recognize, '/recognize')

        self._face_properties_services = _get_and_wait_for_services([
            'keras'
        ], GetFaceProperties, '/get_face_properties')

        self._colour_extractor_services = _get_and_wait_for_services([
            'extract_colour'
        ], ExtractColour, '/extract_colour')

        self._bridge = CvBridge()

        rospy.loginfo("People detector initialized")

    def _get_recognitions(self, img):
        """
        Get recognitions from openpose and openface
        :param: Input image (as cv image) received by people detector service
        """
        args = zip(self._recognize_services.values(), [{
            "image": self._bridge.cv2_to_imgmsg(img, "bgr8")
        }] * len(self._recognize_services))

        with closing(Pool(len(self._recognize_services))) as p:  # Without closing we have a memory leak
            return dict(zip(self._recognize_services.keys(), p.map(_threaded_srv, args)))

    def _get_face_properties(self, images):
        """
        Get face properties from Keras
        :param: face images as cv images
        """
        args = zip(self._face_properties_services.values(), [{
            "face_image_array": [self._bridge.cv2_to_imgmsg(image, "bgr8") for image in images]
        }] * len(self._face_properties_services))

        with closing(Pool(len(self._face_properties_services))) as p:  # Without closing we have a memory leak
            result = dict(zip(self._face_properties_services.keys(), p.map(_threaded_srv, args)))

        return result['keras'].properties_array

    def _get_colour_extractor(self, img):
        """
        Get results of the colour extractor service
        :param: CV image to extract colour from
        """
        args = zip(self._colour_extractor_services.values(), [{
            "image": self._bridge.cv2_to_imgmsg(img, "bgr8")
        }] * len(self._colour_extractor_services))

        with closing(Pool(len(self._colour_extractor_services))) as p:
            result = dict(zip(self._colour_extractor_services.keys(), p.map(_threaded_srv, args)))

        return result


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
        return body_parts = [r for r in recognitions if r.group_id == group_id]

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
        label = " shirt colours:"
        for colour in shirt_colours['extract_colour'].colours:
            label += " {}".format(colour.data)
        return label

    @staticmethod
    def move_face_roi_to_shirt(face_roi, img):
        """
        Given a ROI for a face, shift the ROI to the person's shirt. Assuming the person is upright :/
        :param face_roi: RegionOfInterest
        :param img: cv2 image
        :return: RegionOfInterest
        """
        shirt_roi = copy.deepcopy(face_roi)
        shirt_roi.height = face_roi.height
        shirt_roi.y_offset += int(face_roi.height * 1.5)
        shirt_roi.y_offset = min(shirt_roi.y_offset, img.shape[0] - shirt_roi.height)
        rospy.logdebug("face_roi: {}, shirt_roi: {}, img.shape: {}".format(face_roi, shirt_roi, img.shape))
        return shirt_roi

    def recognize(self, image):
        # OpenPose and OpenFace service calls
        start_recognize = time.time()
        recognitions = self._get_recognitions(image)
        rospy.logdebug("Recognize took %.4f seconds", time.time() - start_recognize)

        # Extract face ROIs and their corresponding group ids from recognitions of openpose
        openpose_face_rois, openpose_face_group_ids = PeopleDetector._get_face_rois_ids_openpose(recognitions['openpose'].recognitions)

        body_parts_array = [PeopleDetector._get_body_parts_openpose(group_id,
            recognitions['openpose'].recognitions) for group_id in openpose_face_group_ids]

        face_recognitions = [PeopleDetector._get_container_recognition(openpose_face_roi,
                                                                       recognitions['openface'].recognitions)
                             for openpose_face_roi in openpose_face_rois]

        face_labels = [PeopleDetector._get_best_label(r) for r in face_recognitions]
        face_images = [PeopleDetector._image_from_roi(image, r.roi) for r in face_recognitions]

        # Keras service call
        face_properties_array = self._get_face_properties(face_images)

        # Colour Extractor service call
        shirt_images = [PeopleDetector._image_from_roi(image, PeopleDetector.move_face_roi_to_shirt(r.roi, image)) for r in face_recognitions]
        shirt_colours_array = [self._get_colour_extractor(img) for img in shirt_images]

        # Prepare image annotation labels and People message
        image_annotations = list()
        people = list()

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
