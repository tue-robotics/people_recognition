#!/usr/bin/env python

import rospy
import message_filters
from people_tracking_v2.msg import ComparisonScoresArray, DetectionArray
from std_msgs.msg import Int32

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
        self.decision_pub = rospy.Publisher('/decision/result', Int32, queue_size=10)
        
        rospy.spin()
    
    def sync_callback(self, comparison_msg, detection_msg):
        """Callback function to handle synchronized comparison scores and detection info."""
        # Define thresholds
        iou_threshold = 0.8
        hoc_threshold = 0.3
        pose_threshold = 0.5  # Example threshold for pose distance

        valid_detections = []
        iou_detections = []

        # Create a dictionary for quick lookup of IoU values by detection ID
        iou_dict = {detection.id: detection.iou for detection in detection_msg.detections}

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
                        valid_detections.append((score.id, hoc_distance_score))
                else:
                    # Check if both HoC and pose scores are valid
                    if hoc_distance_score < hoc_threshold and pose_distance_score < pose_threshold:
                        valid_detections.append((score.id, hoc_distance_score))

        if valid_detections:
            # Find the detection with the best (lowest) HoC score among the valid detections
            best_hoc_detection = min(valid_detections, key=lambda x: x[1])[0]
            operator_id = best_hoc_detection
            decision_source = "HoC + Pose"
        elif iou_detections:
            # If there are no valid detections but there are high IoU detections, use the highest IoU detection
            best_iou_detection = max(iou_detections, key=lambda x: x[1])[0]
            operator_id = best_iou_detection
            decision_source = "IoU"
        else:
            operator_id = -1  # Use -1 to indicate no operator found
            decision_source = "None"

        # Publish the final decision
        self.decision_pub.publish(operator_id)

        # Log the decision
        decision_log = f"Operator Detection ID {operator_id} determined by {decision_source}"
        rospy.loginfo(decision_log)

if __name__ == '__main__':
    try:
        DecisionNode()
    except rospy.ROSInterruptException:
        pass
