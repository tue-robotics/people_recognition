#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image
from people_tracking_v2.msg import ComparisonScoresArray, DetectionArray, DecisionResult
import os
import csv
from datetime import datetime
import rospkg
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys

laptop = sys.argv[1]
name_subscriber_RGB = 'Webcam/image_raw' if laptop == "True" else '/hero/head_rgbd_sensor/rgb/image_raw'
depth_camera = False if sys.argv[2] == "False" else True
save_data = False if sys.argv[3] == "False" else True

class DecisionNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('decision_node', anonymous=True)

        # Subscribers for comparison scores array, detection info, and RGB image
        comparison_sub = message_filters.Subscriber('/comparison/scores_array', ComparisonScoresArray)
        detection_sub = message_filters.Subscriber('/detections_info', DetectionArray)
        image_sub = message_filters.Subscriber(name_subscriber_RGB, Image)

        # Synchronize the subscribers
        ts = message_filters.ApproximateTimeSynchronizer([comparison_sub, detection_sub, image_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.sync_callback)

        # Publisher for decision results and marked images
        self.decision_pub = rospy.Publisher('/decision/result', DecisionResult, queue_size=10)
        self.marked_image_pub = rospy.Publisher('/marked_image', Image, queue_size=10)

        # Initialize variables for saving data
        self.save_data = save_data
        if self.save_data:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path("people_tracking_v2")
            date_str = datetime.now().strftime('%a %b %d Test case 4')
            self.save_path = os.path.join(package_path, f'data/Excel {date_str}/')
            os.makedirs(self.save_path, exist_ok=True)
            self.csv_file_path = os.path.join(self.save_path, 'decision_data.csv')
            self.marked_image_save_path = os.path.join(self.save_path, 'marked_images/')
            os.makedirs(self.marked_image_save_path, exist_ok=True)
            self.init_csv_file()
        
        self.bridge = CvBridge()
        self.image_counter = 0

        rospy.spin()

    def init_csv_file(self):
        """Initialize the CSV file with headers."""
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Operator ID', 'Decision Source', 'HoC Scores', 'Pose Scores', 'IoU Values'])

    def sync_callback(self, comparison_msg, detection_msg, image_msg):
        """Callback function to handle synchronized comparison scores, detection info, and RGB image."""
        # Define thresholds
        iou_threshold = 0.8
        hoc_threshold = 0.45
        pose_threshold = 0.1

        iou_detections = []
        hoc_pose_detections = []

        # Create a dictionary for quick lookup of IoU values and HoC values by detection ID
        iou_dict = {detection.id: detection.iou for detection in detection_msg.detections}
        hoc_dict = {score.id: score.hoc_distance_score for score in comparison_msg.scores}
        pose_dict = {score.id: score.pose_distance_score for score in comparison_msg.scores}

        # Check for detections with high IoU values
        for detection in detection_msg.detections:
            if detection.iou > iou_threshold:
                iou_detections.append((detection.id, detection.iou))

        # Iterate over each comparison score in the array
        for score in comparison_msg.scores:
            hoc_distance_score = score.hoc_distance_score
            pose_distance_score = score.pose_distance_score

            # Get the IoU value from the synchronized detection message
            iou = iou_dict.get(score.id, 0.0)

            if iou <= iou_threshold:
                # If pose distance score is negative, only consider HoC score
                if pose_distance_score < 0:
                    if hoc_distance_score < hoc_threshold:
                        hoc_pose_detections.append((score.id, hoc_distance_score))
                else:
                    # Check if both HoC and pose scores are valid
                    if hoc_distance_score < hoc_threshold and pose_distance_score < pose_threshold:
                        hoc_pose_detections.append((score.id, hoc_distance_score))

        if iou_detections:
            # Find the detection with the highest IoU
            best_iou_detection = max(iou_detections, key=lambda x: x[1])[0]

            # Check if there is any hoc/pose passing detection
            if hoc_pose_detections:
                # Find the detection with the smallest HoC score among the hoc_pose_detections
                best_hoc_detection = min(hoc_pose_detections, key=lambda x: x[1])[0]

                # Extract HoC values for comparison
                best_iou_hoc_value = hoc_dict.get(best_iou_detection, float('inf'))
                best_hoc_value = hoc_dict.get(best_hoc_detection, float('inf'))

                # Compare the HoC values of best_iou_detection and best_hoc_detection
                if best_hoc_value < best_iou_hoc_value:
                    operator_id = best_hoc_detection
                    decision_source = "HoC + Pose"
                else:
                    operator_id = best_iou_detection
                    decision_source = "IoU"
            else:
                operator_id = best_iou_detection
                decision_source = "IoU"
        else:
            # If there are no IoU detections, look at detections that pass HoC and pose thresholds
            if hoc_pose_detections:
                best_hoc_detection = min(hoc_pose_detections, key=lambda x: x[1])[0]
                operator_id = best_hoc_detection
                decision_source = "HoC + Pose to start IoU"
            else:
                operator_id = -1  # Use -1 to indicate no operator found
                decision_source = "None"

        # Publish the final decision
        decision_result = DecisionResult()
        decision_result.operator_id = operator_id
        decision_result.decision_source = decision_source
        self.decision_pub.publish(decision_result)

        # Log the decision
        decision_log = f"Operator Detection ID {operator_id} determined by {decision_source}"
        rospy.loginfo(decision_log)

        # Save the data to CSV
        if self.save_data:
            self.save_to_csv(operator_id, decision_source, comparison_msg, detection_msg)

        # Process and publish marked image
        self.process_and_publish_image(image_msg, detection_msg, operator_id)

    def save_to_csv(self, operator_id, decision_source, comparison_msg, detection_msg):
        """Save the required data to CSV."""
        hoc_scores = [score.hoc_distance_score for score in comparison_msg.scores]
        pose_scores = [score.pose_distance_score for score in comparison_msg.scores]
        iou_values = [detection.iou for detection in detection_msg.detections]

        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([operator_id, decision_source, hoc_scores, pose_scores, iou_values])

    def process_and_publish_image(self, image_msg, detection_msg, operator_id):
        """Process the RGB image to mark the operator and publish the marked image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        # Mark the operator with a red dot if found
        if operator_id > 0:
            for detection in detection_msg.detections:
                if detection.id == operator_id:
                    x_center = int((detection.x1 + detection.x2) / 2)
                    y_center = int((detection.y1 + detection.y2) / 2)
                    cv2.circle(cv_image, (x_center, y_center), 5, (0, 0, 255), -1)
                    break

        # Convert the image back to ROS message
        try:
            marked_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            marked_image_msg.header.stamp = image_msg.header.stamp
            self.marked_image_pub.publish(marked_image_msg)

            # Save the image with the red dot
            if self.save_data:
                image_save_path = os.path.join(self.marked_image_save_path, f"{self.image_counter}.png")
                cv2.imwrite(image_save_path, cv_image)
                self.image_counter += 1
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error while publishing: {e}")

if __name__ == '__main__':
    try:
        DecisionNode()
    except rospy.ROSInterruptException:
        pass
