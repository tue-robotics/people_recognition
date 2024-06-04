#!/usr/bin/env python

import rospy
from people_tracking_v2.msg import ComparisonScores
from std_msgs.msg import String

class DecisionNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('decision_node', anonymous=True)
        
        # Subscriber for comparison scores
        self.comparison_sub = rospy.Subscriber('/comparison/scores', ComparisonScores, self.comparison_callback)
        
        # Publisher for decision results
        self.decision_pub = rospy.Publisher('/decision/result', String, queue_size=10)
        
        rospy.spin()
    
    def comparison_callback(self, msg):
        """Callback function to handle comparison scores and make the final decision."""
        hoc_distance_score = msg.hoc_distance_score
        pose_distance_score = msg.pose_distance_score

        # Define thresholds
        hoc_threshold = 0.1
        pose_threshold = 10.0

        # Logical structure for final decision
        if hoc_distance_score < hoc_threshold and pose_distance_score < pose_threshold:
            is_operator = True
        else:
            is_operator = False

        # Publish the final decision
        decision_msg = String()
        decision_msg.data = f"Detection ID {msg.id}: Is known operator: {is_operator}"
        self.decision_pub.publish(decision_msg)
        #rospy.loginfo(decision_msg.data)

if __name__ == '__main__':
    try:
        DecisionNode()
    except rospy.ROSInterruptException:
        pass
