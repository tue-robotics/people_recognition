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
        iou_threshold = 0.9
        hoc_threshold = 0.6

        iou_detections = []
        best_hoc_detection = None
        best_hoc_score = float('inf')

        # Create a dictionary for quick lookup of IoU values by detection ID
        iou_dict = {detection.id: detection.iou for detection in detection_msg.detections}

        # Iterate over each comparison score in the array
        for score in comparison_msg.scores:
            hoc_distance_score = score.hoc_distance_score
            pose_distance_score = score.pose_distance_score

            # Get the IoU value from the synchronized detection message
            iou = iou_dict.get(score.id, 0.0)

            # Check if IoU is over the threshold
            if iou > iou_threshold:
                iou_detections.append(score.id)

            # Check if HoC score is under the threshold
            if hoc_distance_score < hoc_threshold:
                if hoc_distance_score < best_hoc_score:
                    best_hoc_score = hoc_distance_score
                    best_hoc_detection = score.id

        if len(iou_detections) == 1:
            operator_id = iou_detections[0]
            decision_source = "IoU"
        elif best_hoc_detection is not None:
            operator_id = best_hoc_detection
            decision_source = "HoC"
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
