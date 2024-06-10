#!/usr/bin/env python

import rospy
from people_tracking_v2.msg import DetectionArray, BodySize, BodySizeArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32  # Import the Float32 message type
import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'people_tracking'))
from people_tracking.yolo_pose_wrapper import YoloPoseWrapper

class PoseEstimationNode:
    def __init__(self, model_name="yolov8n-pose.pt", device="cuda:0"):
        rospy.init_node('pose_estimation', anonymous=True)
        
        self.bridge = CvBridge()
        self._wrapper = YoloPoseWrapper(model_name, device)

        self.detection_sub = rospy.Subscriber("/detections_info", DetectionArray, self.detection_callback)
        self.pose_pub = rospy.Publisher("/pose_distances", BodySizeArray, queue_size=10)
        self.image_sub = rospy.Subscriber("/bounding_box_image", Image, self.image_callback)
        self._result_image_publisher = rospy.Publisher("pose_result", Image, queue_size=10)
        self.iou_threshold = 0.8  # Default threshold value

        self.current_detections = []

    def detection_callback(self, msg):
        self.current_detections = msg.detections

    def image_callback(self, image_msg):
        # Check IoU values and decide whether to process
        should_process = all(detection.iou <= self.iou_threshold for detection in self.current_detections)
        if not should_process:
            rospy.loginfo("Skipping processing due to high IoU value with operator")
            return

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        recognitions, result_image, pose_details = self._wrapper.detect_poses(cv_image)

        pose_distance_array = BodySizeArray()
        pose_distance_array.header.stamp = image_msg.header.stamp

        for pose in pose_details:
            try:
                pose_distance_msg = BodySize()

                # Extract the ID from the incoming message
                detection_id = pose_distance_msg.id

                pose_distance_msg.header.stamp = image_msg.header.stamp  # Use the timestamp from the incoming YOLO image

                if "LEar" in pose and "LAnkle" in pose:
                    left_ear_to_left_ankle = self._wrapper.compute_distance(pose["LEar"], pose["LAnkle"])
                else:
                    left_ear_to_left_ankle = float('inf')

                if "LEar" in pose and "RAnkle" in pose:
                    left_ear_to_right_ankle = self._wrapper.compute_distance(pose["LEar"], pose["RAnkle"])
                else:
                    left_ear_to_right_ankle = float('inf')

                if "REar" in pose and "LAnkle" in pose:
                    right_ear_to_left_ankle = self._wrapper.compute_distance(pose["REar"], pose["LAnkle"])
                else:
                    right_ear_to_left_ankle = float('inf')

                if "REar" in pose and "RAnkle" in pose:
                    right_ear_to_right_ankle = self._wrapper.compute_distance(pose["REar"], pose["RAnkle"])
                else:
                    right_ear_to_right_ankle = float('inf')

                distances = [left_ear_to_left_ankle, left_ear_to_right_ankle, right_ear_to_left_ankle, right_ear_to_right_ankle]
                min_distance = min(distances)

                if min_distance == float('inf'):
                    rospy.logwarn("No valid ear-to-ankle distance found")
                else:
                    pose_distance_msg.head_feet_distance = min_distance
                    rospy.loginfo(f"For ID {detection_id} Head-Feet Distance: {pose_distance_msg.head_feet_distance:.2f}")


                # Find the corresponding detection ID and use depth value to normalize the size
                for detection in self.current_detections:
                    if self.is_pose_within_detection(pose, detection):
                        pose_distance_msg.id = detection.id
                        depth = detection.depth
                        pose_distance_msg.head_feet_distance = self.normalize_size(pose_distance_msg.head_feet_distance, depth)
                        break

                pose_distance_array.distances.append(pose_distance_msg)

                # Draw the pose on the image
                result_image = self.draw_pose(result_image, pose)

            except Exception as e:
                rospy.logerr(f"Error computing distance: {e}")

        # Publish the pose distances as a batch
        self.pose_pub.publish(pose_distance_array)

        # Convert the result image to ROS Image message and publish
        try:
            result_image_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            self._result_image_publisher.publish(result_image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")

    def is_pose_within_detection(self, pose, detection):
        """Check if the pose is within the detection bounding box."""
        x_center = (detection.x1 + detection.x2) / 2
        y_center = (detection.y1 + detection.y2) / 2

        if detection.x1 <= x_center <= detection.x2 and detection.y1 <= y_center <= detection.y2:
            return True
        return False

    def normalize_size(self, size, depth):
        """Normalize the size using the depth value."""
        reference_depth = 1.0  # Use a reference depth (e.g., 1 meter)
        if depth > 0:
            normalized_size = size * (reference_depth / depth)
        else:
            normalized_size = size  # If depth is not available, use the original size
        return normalized_size

    def draw_pose(self, image, pose):
        """Draw the pose on the image."""
        for part in pose:
            x, y = pose[part]
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
        return image

if __name__ == "__main__":
    try:
        node = PoseEstimationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
