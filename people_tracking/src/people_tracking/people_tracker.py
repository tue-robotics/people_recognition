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


class PeopleTracker(object):
    def __init__(self, openpose_srv_name):

        self._openpose_srv = _get_and_wait_for_service(openpose_srv_name, Recognize)

        self._bridge = CvBridge()

        rospy.loginfo("People tracker initialized")

    def run():
        # get image from the camera

        # call detectpeople2D with that image.

        # do tracking...

        # publish results


    
    @staticmethod
    def _image_from_roi(image, roi):
        # ROI needs to be at least 1 pixel in size
        return image[roi.y_offset:roi.y_offset + max(roi.height, 1),
               roi.x_offset:roi.x_offset + max(roi.width, 1)]

    def recognize(self, image_msg):
        assert isinstance(image_msg, Image)
        image = self._bridge.imgmsg_to_cv2(image_msg)

        image_annotations = []
        people = []

        # OpenPose and OpenFace service calls
        rospy.loginfo("Starting pose and face recognition...")
        start_recognize = time.time()
        openpose_response = _get_service_response(self._openpose_srv, image_msg)
        rospy.logdebug("Recognize took %.4f seconds", time.time() - start_recognize)
        rospy.loginfo("_get_face_rois_ids_openpose...")
