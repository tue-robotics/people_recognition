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
        self.operator_iou = 0.0
        self.iou_threshold = 0.9
        self.current_mode = "YOLO_ONLY"
        
        rospy.spin()
    
    def detection_callback(self, msg):
        """Callback function to handle YOLO detections."""
        for detection in msg.detections:
            if detection.id == self.operator_id:
                self.operator_iou = detection.iou  # Assuming IoU is a field in the detection message
                if self.operator_iou > self.iou_threshold:
                    self.set_mode("YOLO_ONLY")
                    return
                else:
                    self.set_mode("YOLO_HOC_POSE")
                    return
        self.set_mode("YOLO_HOC_POSE")
    
    def comparison_callback(self, msg):
        """Callback function to handle comparison scores and make the final decision."""
        hoc_distance_score = msg.hoc_distance_score
        pose_distance_score = msg.pose_distance_score

        hoc_threshold = 0.1
        pose_threshold = 10.0

        if hoc_distance_score < hoc_threshold and pose_distance_score < pose_threshold:
            self.operator_id = msg.id
        else:
            self.operator_id = None

    def set_mode(self, mode):
        """Set the current mode and publish it."""
        if self.current_mode != mode:
            self.current_mode = mode
            mode_msg = String()
            mode_msg.data = f"Current Mode: {self.current_mode}"
            self.mode_pub.publish(mode_msg)
            rospy.loginfo(mode_msg.data)

if __name__ == '__main__':
    try:
        CentralNode()
    except rospy.ROSInterruptException:
        pass
