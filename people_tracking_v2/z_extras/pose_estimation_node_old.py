#!/usr/bin/env python

import os
import socket
import sys
from queue import Empty, Queue

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

import diagnostic_updater
import rospy
from cv_bridge import CvBridge, CvBridgeError
from image_recognition_msgs.msg import Recognitions
from image_recognition_msgs.srv import Recognize
from image_recognition_util import image_writer
from sensor_msgs.msg import Image
from std_msgs.msg import String
from people_tracking.yolo_pose_wrapper import YoloPoseWrapper
from people_tracking_v2.msg import DetectionArray, BodySize

class PoseEstimationNode:
    def __init__(
        self,
        model_name: str,
        device: str,
        conf: float,
        topic_save_images: bool,
        service_save_images: bool,
        topic_publish_result_image: bool,
        service_publish_result_image: bool,
        save_images_folder: str,
    ):
        """
        Openpose node that wraps the openpose functionality and exposes service and subscriber interfaces

        :param model_name: Model to use
        :param device: Device to use
        :param topic_save_images: Whether we would like to store the (result) images that we receive over topics
        :param service_save_images: Whether we would like to store the (result) images that we receive over topics
        :param topic_publish_result_image: Whether we would like to publish the result images of a topic request
        :param service_publish_result_image: Whether we would like to publish the result images of a service request
        :param save_images_folder: Where to store the images
        """
        self._wrapper = YoloPoseWrapper(model_name, device)

        # We need this q construction because openpose python is not thread safe and the rospy client side library
        # uses a thread per pub/sub and service. Since the openpose wrapper is created in the main thread, we have
        # to communicate our openpose requests (inputs) to the main thread where the request is processed by the
        # openpose wrapper (step 1).
        # We have a separate spin loop in the main thread that checks whether there are items in the input q and
        # processes these using the Openpose wrapper (step 2).
        # When the processing has finished, we add the result in the corresponding output queue (specified by the
        # request in the input queue) (step 3).
        self._input_q = Queue()  # image_msg, save_images, publish_images, is_service_request
        self._service_output_q = Queue()  # recognitions
        self._subscriber_output_q = Queue()  # recognitions

        # Debug
        self._topic_save_images = topic_save_images
        self._service_save_images = service_save_images
        self._topic_publish_result_image = topic_publish_result_image
        self._service_publish_result_image = service_publish_result_image
        self._save_images_folder = save_images_folder

        # ROS IO
        self._bridge = CvBridge()
        self._recognize_srv = rospy.Service("recognize", Recognize, self._recognize_srv)
        self._image_subscriber = rospy.Subscriber("/bounding_box_image", Image, self._image_callback)
        self._mode_sub = rospy.Subscriber('/central/mode', String, self.mode_callback)
        self._recognitions_publisher = rospy.Publisher("/pose_recognitions", Recognitions, queue_size=10)
        self._pose_distance_publisher = rospy.Publisher("/pose_distances", BodySize, queue_size=10)
        self._detection_subscriber = rospy.Subscriber("/hero/predicted_detections", DetectionArray, self.detection_callback)  # Add this subscriber
        if self._topic_publish_result_image or self._service_publish_result_image:
            self._result_image_publisher = rospy.Publisher("/pose_result_image", Image, queue_size=10)

        self.last_master_check = rospy.get_time()
        self.current_mode = "YOLO_HOC_POSE"

        rospy.loginfo("PoseEstimationNode initialized:")
        rospy.loginfo(f" - {model_name=}")
        rospy.loginfo(f" - {device=}")
        rospy.loginfo(f" - {conf=}")
        rospy.loginfo(f" - {topic_save_images=}")
        rospy.loginfo(f" - {service_save_images=}")
        rospy.loginfo(f" - {topic_publish_result_image=}")
        rospy.loginfo(f" - {service_publish_result_image=}")
        rospy.loginfo(f" - {save_images_folder=}")

        self.current_detections = []

    def mode_callback(self, msg):
        """Callback to update the current mode."""
        self.current_mode = msg.data.split(": ")[1]

    def detection_callback(self, msg):
        if self.current_mode != "YOLO_HOC_POSE":
            return  # Skip processing if the current mode is not YOLO_HOC_POSE

        #rospy.loginfo(f"First detection received at: {rospy.Time.now()}")  # Log first message timestamp
        """Callback function to handle new detections from YOLO (DetectionArray)."""
        self.current_detections = msg.detections

    def _image_callback(self, image_msg):
        if self.current_mode != "YOLO_HOC_POSE":
            return  # Skip processing if the current mode is not YOLO_HOC_POSE

        self._input_q.put((image_msg, self._topic_save_images, self._topic_publish_result_image, False))
        recognitions, result_image, pose_details = self._wrapper.detect_poses(self._bridge.imgmsg_to_cv2(image_msg, "bgr8"))

        # Calculate distances and publish them
        for pose in pose_details:
            try:
                pose_distance_msg = BodySize()
                pose_distance_msg.header.stamp = image_msg.header.stamp  # Use the timestamp from the incoming YOLO image
                if "LShoulder" in pose and "LHip" in pose:
                    pose_distance_msg.left_shoulder_hip_distance = self._wrapper.compute_distance(pose["LShoulder"], pose["LHip"])
                    #rospy.loginfo(f"Left Shoulder-Hip Distance: {pose_distance_msg.left_shoulder_hip_distance:.2f}")

                if "RShoulder" in pose and "RHip" in pose:
                    pose_distance_msg.right_shoulder_hip_distance = self._wrapper.compute_distance(pose["RShoulder"], pose["RHip"])
                    #rospy.loginfo(f"Right Shoulder-Hip Distance: {pose_distance_msg.right_shoulder_hip_distance:.2f}")

                # Find the corresponding detection ID
                for detection in self.current_detections:
                    if self.is_pose_within_detection(pose, detection):
                        pose_distance_msg.id = detection.id
                        break

                self._pose_distance_publisher.publish(pose_distance_msg)
            except Exception as e:
                rospy.logerr(f"Error computing distance: {e}")

        self._recognitions_publisher.publish(
            Recognitions(header=image_msg.header, recognitions=self._subscriber_output_q.get())
        )

    def _recognize_srv(self, req):
        self._input_q.put((req.image, self._service_save_images, self._service_publish_result_image, True))
        return {"recognitions": self._service_output_q.get()}

    def _get_recognitions(self, image_msg, save_images, publish_images):
        """
        Handles the recognition and publishes and stores the debug images (should be called in the main thread)

        :param image_msg: Incoming image
        :param save_images: Whether to store the images
        :param publish_images: Whether to publish the images
        :return: The recognized recognitions
        """
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert to opencv image: {e}")
            return []

        recognitions, result_image, pose_details = self._wrapper.detect_poses(bgr_image)

        # Log the number of poses detected
        #if recognitions:
            #rospy.loginfo(f"Detected {len(recognitions)} poses")
        #else:
            #rospy.loginfo("No poses detected")

        # Write images
        if save_images:
            image_writer.write_raw(self._save_images_folder, bgr_image)
            image_writer.write_raw(self._save_images_folder, result_image, "overlayed")

        if publish_images:
            self._result_image_publisher.publish(self._bridge.cv2_to_imgmsg(result_image, "bgr8"))

        # Calculate distances and log them
        for pose in pose_details:
            try:
                pose_distance_msg = BodySize()
                pose_distance_msg.header.stamp = rospy.Time.now()
                if "LShoulder" in pose and "LHip" in pose:
                    pose_distance_msg.left_shoulder_hip_distance = self._wrapper.compute_distance(pose["LShoulder"], pose["LHip"])
                    #rospy.loginfo(f"Left Shoulder-Hip Distance: {pose_distance_msg.left_shoulder_hip_distance:.2f}")

                if "RShoulder" in pose and "RHip" in pose:
                    pose_distance_msg.right_shoulder_hip_distance = self._wrapper.compute_distance(pose["RShoulder"], pose["RHip"])
                    #rospy.loginfo(f"Right Shoulder-Hip Distance: {pose_distance_msg.right_shoulder_hip_distance:.2f}")

                # Find the corresponding detection ID
                for detection in self.current_detections:
                    if self.is_pose_within_detection(pose, detection):
                        pose_distance_msg.id = detection.id
                        break

                self._pose_distance_publisher.publish(pose_distance_msg)
            except Exception as e:
                rospy.logerr(f"Error computing distance: {e}")

        return recognitions

    def is_pose_within_detection(self, pose, detection):
        """Check if the pose is within the detection bounding box."""
        x_center = (detection.x1 + detection.x2) / 2
        y_center = (detection.y1 + detection.y2) / 2

        if detection.x1 <= x_center <= detection.x2 and detection.y1 <= y_center <= detection.y2:
            return True
        return False

    def spin(self, check_master: bool = False):
        """
        Empty input queues and fill output queues (see __init__ doc)
        """
        while not rospy.is_shutdown():
            try:
                image_msg, save_images, publish_images, is_service_request = self._input_q.get(timeout=1.0)
            except Empty:
                pass
            else:
                if is_service_request:
                    self._service_output_q.put(self._get_recognitions(image_msg, save_images, publish_images))
                else:
                    self._subscriber_output_q.put(self._get_recognitions(image_msg, save_images, publish_images))
            finally:
                if check_master and rospy.get_time() >= self.last_master_check + 1:
                    self.last_master_check = rospy.get_time()
                    try:
                        rospy.get_master().getPid()
                    except socket.error:
                        rospy.logdebug("Connection to master is lost")
                        return 1  # This should result in a non-zero error code of the entire program

        return 0

if __name__ == "__main__":
    rospy.init_node("pose_estimation")

    try:
        node = PoseEstimationNode(
            rospy.get_param("~model", "yolov8n-pose.pt"),
            rospy.get_param("~device", "cuda:0"),
            rospy.get_param("~conf", 0.25),
            rospy.get_param("~topic_save_images", False),
            rospy.get_param("~service_save_images", True),
            rospy.get_param("~topic_publish_result_image", True),
            rospy.get_param("~service_publish_result_image", True),
            os.path.expanduser(rospy.get_param("~save_images_folder", "/tmp/pose_estimation")),
        )

        check_master: bool = rospy.get_param("~check_master", False)

        updater = diagnostic_updater.Updater()
        updater.setHardwareID("none")
        updater.add(diagnostic_updater.Heartbeat())
        rospy.Timer(rospy.Duration(1), lambda event: updater.force_update())

        sys.exit(node.spin(check_master))
    except Exception as e:
        rospy.logfatal(e)
        raise
