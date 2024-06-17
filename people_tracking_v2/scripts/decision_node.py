#!/usr/bin/env python

import rospy
import message_filters
from people_tracking_v2.msg import ComparisonScoresArray, DetectionArray, DecisionResult

class DecisionNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('decision_node', anonymous=True)
        
        # Subscribers for comparison scores array and detection info
        comparison_sub = message_filters.Subscriber('/comparison/scores_array', ComparisonScoresArray)
        detection_sub = message_filters.Subscriber('/detections_info', DetectionArray)
        
        # Synchronize the subscribers
        ts = message_filters.ApproximateTimeSynchronizer([comparison_sub, detection_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.sync_callback)
        
        # Publisher for decision results
        self.decision_pub = rospy.Publisher('/decision/result', DecisionResult, queue_size=10)
        
        rospy.spin()
    
    def sync_callback(self, comparison_msg, detection_msg):
        """Callback function to handle synchronized comparison scores and detection info."""
        # Define thresholds
        iou_threshold = 0.8
        hoc_threshold = 0.3
        pose_threshold = 0.5  # Example threshold for pose distance

        iou_detections = []
        hoc_pose_detections = []

        # Create a dictionary for quick lookup of IoU values and HoC values by detection ID
        iou_dict = {detection.id: detection.iou for detection in detection_msg.detections}
        hoc_dict = {score.id: score.hoc_distance_score for score in comparison_msg.scores}

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

if __name__ == '__main__':
    try:
        DecisionNode()
    except rospy.ROSInterruptException:
        pass
