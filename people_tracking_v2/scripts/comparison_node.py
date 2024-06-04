#!/usr/bin/env python

import rospy
import numpy as np
from people_tracking_v2.msg import HoCVectorArray, BodySizeArray, ComparisonScores, ComparisonScoresArray  # Import the custom message types
from std_msgs.msg import String
import os

class ComparisonNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('comparison_node', anonymous=True)
        
        # Subscribers for batch messages
        self.hoc_sub = rospy.Subscriber('/hoc_vectors_array', HoCVectorArray, self.hoc_callback)
        self.pose_sub = rospy.Subscriber('/pose_distances_array', BodySizeArray, self.pose_callback)
        
        # Publisher for comparison scores batch
        self.comparison_pub = rospy.Publisher('/comparison/scores_array', ComparisonScoresArray, queue_size=10)
        
        # Load saved HoC and Pose data
        self.hoc_data_file = os.path.expanduser('~/hoc_data/hoc_data.npz')
        self.pose_data_file = os.path.expanduser('~/pose_data/pose_data.npz')
        self.load_hoc_data()
        self.load_pose_data()

        self.hoc_data = {}
        self.pose_data = {}
        
        rospy.spin()
    
    def load_hoc_data(self):
        """Load the saved HoC data from the .npz file (HoC)."""
        if os.path.exists(self.hoc_data_file):
            data = np.load(self.hoc_data_file)
            self.saved_hue = data['hue'][0]
            self.saved_sat = data['sat'][0]
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
        else:
            rospy.logerr(f"Pose data file {self.pose_data_file} not found")
            self.saved_pose_data = None

    def hoc_callback(self, msg):
        """Callback function to handle batch HoC data."""
        for vector in msg.vectors:
            self.hoc_data[vector.id] = vector
        self.compare_and_publish()

    def pose_callback(self, msg):
        """Callback function to handle batch pose data."""
        for distance in msg.distances:
            self.pose_data[distance.id] = distance
        self.compare_and_publish()

    def compare_and_publish(self):
        """Compare HoC and pose data for detections present in both batches and publish results."""
        if not self.hoc_data or not self.pose_data:
            return
        
        comparison_scores_array = ComparisonScoresArray()
        comparison_scores_array.header.stamp = rospy.Time.now()

        for detection_id in self.hoc_data.keys() & self.pose_data.keys():
            hoc_vector = self.hoc_data[detection_id]
            pose_distance = self.pose_data[detection_id]

            # Compare HoC data
            hue_vector = np.array(hoc_vector.hue_vector)
            sat_vector = np.array(hoc_vector.sat_vector)
            hoc_distance_score = self.compute_hoc_distance_score(hue_vector, sat_vector)

            # Compare pose data
            left_shoulder_hip_distance = pose_distance.left_shoulder_hip_distance
            right_shoulder_hip_distance = pose_distance.right_shoulder_hip_distance
            left_shoulder_hip_saved = np.mean(self.saved_pose_data['left_shoulder_hip_distance'])
            right_shoulder_hip_saved = np.mean(self.saved_pose_data['right_shoulder_hip_distance'])

            left_distance = self.compute_distance(left_shoulder_hip_distance, left_shoulder_hip_saved)
            right_distance = self.compute_distance(right_shoulder_hip_distance, right_shoulder_hip_saved)
            pose_distance_score = (left_distance + right_distance) / 2

            # Create and append comparison score
            comparison_scores_msg = ComparisonScores()
            comparison_scores_msg.id = detection_id
            comparison_scores_msg.hoc_distance_score = hoc_distance_score
            comparison_scores_msg.pose_distance_score = pose_distance_score
            comparison_scores_array.scores.append(comparison_scores_msg)

        self.comparison_pub.publish(comparison_scores_array)

    def compute_hoc_distance_score(self, hue_vector, sat_vector):
        """Compute the distance score between the current detection and saved data (HoC)."""
        hue_distance = self.compute_distance(hue_vector, self.saved_hue)
        sat_distance = self.compute_distance(sat_vector, self.saved_sat)
        return (hue_distance + sat_distance) / 2
    
    def compute_distance(self, vector1, vector2):
        """Compute the Euclidean distance between two vectors (General)."""
        return np.linalg.norm(vector1 - vector2)

if __name__ == '__main__':
    try:
        ComparisonNode()
    except rospy.ROSInterruptException:
        pass
