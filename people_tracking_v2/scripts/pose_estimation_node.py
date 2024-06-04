#!/usr/bin/env python

import rospy
from people_tracking_v2.msg import DetectionArray, BodySize, BodySizeArray  # Custom message for batch pose distances
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import sys
from queue import Empty, Queue

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from people_tracking.yolo_pose_wrapper import YoloPoseWrapper

class PoseEstimationNode:
    def __init__(self):
        rospy.init_node('pose_estimation_node', anonymous=True)
        
        self.bridge = CvBridge()
        self._wrapper = YoloPoseWrapper(model_name="yolov8n-pose.pt", device="cuda:0")
        
        self.image_sub = rospy.Subscriber("/bounding_box_image", Image, self.image_callback)
        self.pose_distance_pub = rospy.Publisher("/pose_distances", BodySizeArray, queue_size=10)
        self.detection_sub = rospy.Subscriber("/detections", DetectionArray, self.detection_callback)
        
        self.current_detections = []
        rospy.spin()
    
    def detection_callback(self, msg):
        """Callback function to handle new detections from YOLO (DetectionArray)."""
        self.current_detections = msg.detections

    def image_callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            recognitions, result_image, pose_details = self._wrapper.detect_poses(cv_image)

            pose_distance_array = BodySizeArray()
            pose_distance_array.header.stamp = image_msg.header.stamp  # Use the timestamp from the incoming YOLO image

            for pose in pose_details:
                pose_distance_msg = BodySize()
                pose_distance_msg.header.stamp = image_msg.header.stamp  # Use the same timestamp

                if "LShoulder" in pose and "LHip" in pose:
                    pose_distance_msg.left_shoulder_hip_distance = self._wrapper.compute_distance(pose["LShoulder"], pose["LHip"])

                if "RShoulder" in pose and "RHip" in pose:
                    pose_distance_msg.right_shoulder_hip_distance = self._wrapper.compute_distance(pose["RShoulder"], pose["RHip"])

                for detection in self.current_detections:
                    if self.is_pose_within_detection(pose, detection):
                        pose_distance_msg.id = detection.id
                        break

                pose_distance_array.distances.append(pose_distance_msg)

            self.pose_distance_pub.publish(pose_distance_array)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def is_pose_within_detection(self, pose, detection):
        """Check if the pose is within the detection bounding box."""
        x_center = (detection.x1 + detection.x2) / 2
        y_center = (detection.y1 + detection.y2) / 2

        if detection.x1 <= x_center <= detection.x2 and detection.y1 <= y_center <= detection.y2:
            return True
        return False

if __name__ == '__main__':
    try:
        PoseEstimationNode()
    except rospy.ROSInterruptException:
        pass
