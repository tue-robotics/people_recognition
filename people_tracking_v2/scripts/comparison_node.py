#!/usr/bin/env python

import rospy
import numpy as np
import message_filters
from people_tracking_v2.msg import HoCVector, BodySize, ComparisonScores  # Import the custom message types
from std_msgs.msg import String
import os

class ComparisonNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('comparison_node', anonymous=True)
        
        # Synchronize the subscribers using message_filters
        hoc_sub = message_filters.Subscriber('/hoc_vectors', HoCVector)
        pose_sub = message_filters.Subscriber('/pose_distances', BodySize)
        
        ts = message_filters.ApproximateTimeSynchronizer([hoc_sub, pose_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.sync_callback)
        
        # Publisher for comparison scores
        self.comparison_pub = rospy.Publisher('/comparison/scores', ComparisonScores, queue_size=10)
        
        # Load saved HoC and Pose data
        self.hoc_data_file = os.path.expanduser('~/hoc_data/hoc_data.npz')
        self.pose_data_file = os.path.expanduser('~/pose_data/pose_data.npz')
        self.load_hoc_data()
        self.load_pose_data()
        
        rospy.spin()
    
    def load_hoc_data(self):
        """Load the saved HoC data from the .npz file (HoC)."""
        if os.path.exists(self.hoc_data_file):
            data = np.load(self.hoc_data_file)
            self.saved_hue = data['hue'][0]
            self.saved_sat = data['sat'][0]
            #rospy.loginfo(f"Loaded HoC data from {self.hoc_data_file}")
        else:
            rospy.logerr(f"HoC data file {self.hoc_data_file} not found")
            self.saved_hue = None
            self.saved_sat = None

    def load_pose_data(self):
        """Load the saved Pose data from the .npz file (Pose)."""
        if os.path.exists(self.pose_data_file):
            data = np.load(self.pose_data_file)
            self.saved_pose_data = {
                'left_shoulder_hip_distance': data['left_shoulder_hip_distance'],
                'right_shoulder_hip_distance': data['right_shoulder_hip_distance']
            }
            #rospy.loginfo(f"Loaded Pose data from {self.pose_data_file}")
        else:
            rospy.logerr(f"Pose data file {self.pose_data_file} not found")
            self.saved_pose_data = None

    def sync_callback(self, hoc_msg, pose_msg):
        """Callback function to handle synchronized HoC and pose data."""
        if self.saved_hue is None or self.saved_sat is None:
            rospy.logerr("No saved HoC data available for comparison")
            return

        # Log timestamps and IDs for synchronization verification
        #rospy.loginfo(f"Synchronized messages: HoC timestamp: {hoc_msg.header.stamp}, Pose timestamp: {pose_msg.header.stamp}")
        #rospy.loginfo(f"Detection ID {hoc_msg.id}: HoC and Pose data synchronized")

        # Compare HoC data
        hue_vector = hoc_msg.hue_vector
        sat_vector = hoc_msg.sat_vector
        hoc_distance_score = self.compute_hoc_distance_score(hue_vector, sat_vector)
        #rospy.loginfo(f"Detection ID {hoc_msg.id}: HoC Distance score: {hoc_distance_score:.2f}")

        # Compare pose data
        left_shoulder_hip_distance = pose_msg.left_shoulder_hip_distance
        right_shoulder_hip_distance = pose_msg.right_shoulder_hip_distance
        left_shoulder_hip_saved = np.mean(self.saved_pose_data['left_shoulder_hip_distance'])
        right_shoulder_hip_saved = np.mean(self.saved_pose_data['right_shoulder_hip_distance'])

        left_distance = self.compute_distance(left_shoulder_hip_distance, left_shoulder_hip_saved)
        right_distance = self.compute_distance(right_shoulder_hip_distance, right_shoulder_hip_saved)
        pose_distance_score = (left_distance + right_distance) / 2
        #rospy.loginfo(f"Detection ID {pose_msg.id}: Pose Distance score: {pose_distance_score:.2f}")

        # Publish comparison scores
        comparison_scores_msg = ComparisonScores()
        comparison_scores_msg.header.stamp = rospy.Time.now()
        comparison_scores_msg.id = hoc_msg.id
        comparison_scores_msg.hoc_distance_score = hoc_distance_score
        comparison_scores_msg.pose_distance_score = pose_distance_score
        self.comparison_pub.publish(comparison_scores_msg)

    def compute_hoc_distance_score(self, hue_vector, sat_vector):
        """Compute the distance score between the current detection and saved data (HoC)."""
        hue_vector = np.array(hue_vector)
        sat_vector = np.array(sat_vector)
        
        hue_distance = self.compute_distance(hue_vector, self.saved_hue)
        sat_distance = self.compute_distance(sat_vector, self.saved_sat)
        
        return (hue_distance + sat_distance) / 2
    
    def compute_distance(self, vector1, vector2):
        """Compute the Euclidean distance between two vectors (General)."""
        return np.linalg.norm(vector1 - vector2)
    
    def publish_debug_info(self, hoc_distance_score, pose_distance_score, detection_id):
        """Publish debug information about the current comparison (General)."""
        debug_msg = String()
        debug_msg.data = f"Detection ID {detection_id}: HoC Distance score: {hoc_distance_score:.2f}, Pose Distance score: {pose_distance_score:.2f}"
        self.publisher_debug.publish(debug_msg)

if __name__ == '__main__':
    try:
        ComparisonNode()
    except rospy.ROSInterruptException:
        pass
