#!/usr/bin/env python

import rospy
import numpy as np
import message_filters
from people_tracking_v2.msg import BodySizeArray, ComparisonScoresArray, ComparisonScores  # Import the custom message types
from std_msgs.msg import String
import os
import csv

class PoseComparisonNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('pose_comparison_node', anonymous=True)
        
        # Subscriber for pose distances
        pose_sub = message_filters.Subscriber('/pose_distances', BodySizeArray)
        
        # Synchronize the subscriber using message_filters
        ts = message_filters.ApproximateTimeSynchronizer([pose_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.pose_callback)
        
        # Publisher for comparison scores
        self.comparison_pub = rospy.Publisher('/comparison/pose_scores_array', ComparisonScoresArray, queue_size=10)
        
        # Load saved pose data
        self.pose_data_file = os.path.expanduser('~/pose_data/pose_data.npz')
        self.load_pose_data()
        
        # File to save detection IDs and scores
        self.detection_log_file = os.path.expanduser('~/detection_scores_pose.csv')
        self.init_detection_log_file()
        
        rospy.spin()
    
    def load_pose_data(self):
        """Load the saved pose data from the .npz file."""
        if os.path.exists(self.pose_data_file):
            data = np.load(self.pose_data_file)
            self.saved_pose_data = {
                'head_feet_distance': data['head_feet_distance']
            }
            rospy.loginfo(f"Loaded pose data from {self.pose_data_file}")
        else:
            rospy.logerr(f"Pose data file {self.pose_data_file} not found")
            self.saved_pose_data = None

    def init_detection_log_file(self):
        """Initialize the detection log file."""
        with open(self.detection_log_file, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame Timestamp', 'Detection ID', 'Pose Distance Score'])
        rospy.loginfo(f"Initialized detection log file at {self.detection_log_file}")

    def pose_callback(self, pose_array):
        """Callback function to handle pose data."""
        rospy.loginfo("pose_callback invoked")

        if self.saved_pose_data is None:
            rospy.logerr("No saved pose data available for comparison")
            return

        if not pose_array.distances:
            rospy.logerr("Received empty pose array")
            return

        comparison_scores_array = ComparisonScoresArray()
        comparison_scores_array.header.stamp = pose_array.header.stamp

        for pose_msg in pose_array.distances:
            rospy.loginfo(f"Processing Detection ID {pose_msg.id}")

            # Compare pose data
            head_feet_distance = pose_msg.head_feet_distance
            head_feet_saved = np.mean(self.saved_pose_data['head_feet_distance'])

            distance_score = self.compute_distance(head_feet_distance, head_feet_saved)
            rospy.loginfo(f"Detection ID {pose_msg.id}: Pose Distance score: {distance_score:.2f}")

            # Record each person's score and ID
            self.save_detection_score(pose_array.header.stamp, pose_msg.id, distance_score)

            # Create and append ComparisonScores message
            comparison_scores_msg = ComparisonScores()
            comparison_scores_msg.header.stamp = pose_msg.header.stamp  # Use the timestamp from the pose message
            comparison_scores_msg.header.frame_id = pose_msg.header.frame_id
            comparison_scores_msg.id = pose_msg.id
            comparison_scores_msg.hoc_distance_score = 0.0  # Set to 0.0 since it's not used
            comparison_scores_msg.pose_distance_score = distance_score
            comparison_scores_array.scores.append(comparison_scores_msg)

        # Publish the comparison scores as a batch
        self.comparison_pub.publish(comparison_scores_array)

    def save_detection_score(self, timestamp, detection_id, score):
        """Save the detection ID and score for each frame."""
        with open(self.detection_log_file, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, detection_id, score])
        rospy.loginfo(f"Saved detection ID {detection_id} with score {score} at timestamp {timestamp}")

    def compute_distance(self, distance1, distance2):
        """Compute the Euclidean distance between two scalar values."""
        return np.abs(distance1 - distance2)

if __name__ == '__main__':
    try:
        PoseComparisonNode()
    except rospy.ROSInterruptException:
        pass
