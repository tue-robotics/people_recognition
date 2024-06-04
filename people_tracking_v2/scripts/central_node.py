#!/usr/bin/env python

import rospy
from people_tracking_v2.msg import DetectionArray, ComparisonScores
from std_msgs.msg import String

class CentralNode:
    def __init__(self):
        rospy.init_node('central_node', anonymous=True)
        
        # Subscribers
        self.detection_sub = rospy.Subscriber('/hero/predicted_detections', DetectionArray, self.detection_callback)
        self.comparison_sub = rospy.Subscriber('/comparison/scores', ComparisonScores, self.comparison_callback)
        
        # Publishers
        self.mode_pub = rospy.Publisher('/central/mode', String, queue_size=10)
        
        self.operator_id = None
        self.iou_threshold = 0.9
        self.current_mode = "YOLO_HOC_POSE"  # Define initial mode
        
        rospy.spin()
    
    def detection_callback(self, msg):
        """Callback function to handle YOLO detections."""
        for detection in msg.detections:
            if detection.id == self.operator_id:
                if detection.iou > self.iou_threshold:
                    self.set_mode("YOLO_ONLY")  # Set mode to YOLO_ONLY if IoU is above the threshold
                else:
                    self.set_mode("YOLO_HOC_POSE")  # Set mode to YOLO_HOC_POSE if IoU is below the threshold
                return  # Exit after processing the operator detection
        self.set_mode("YOLO_HOC_POSE")  # If operator is not detected at all
    
    def comparison_callback(self, msg):
        """Callback function to handle comparison scores and update operator ID."""
        hoc_threshold = 0.1
        pose_threshold = 10.0

        if msg.hoc_distance_score < hoc_threshold and msg.pose_distance_score < pose_threshold:
            self.operator_id = msg.id  # Update operator ID if scores are below thresholds
        else:
            self.operator_id = None  # Reset operator ID if scores are above thresholds

    def set_mode(self, mode):
        """Set the current mode and publish it."""
        if self.current_mode != mode:
            self.current_mode = mode  # Update current mode
            mode_msg = String()
            mode_msg.data = f"Current Mode: {self.current_mode}"  # Define mode message
            self.mode_pub.publish(mode_msg)  # Publish mode message to /central/mode topic
            rospy.loginfo(mode_msg.data)  # Log the current mode

if __name__ == '__main__':
    try:
        CentralNode()
    except rospy.ROSInterruptException:
        pass
