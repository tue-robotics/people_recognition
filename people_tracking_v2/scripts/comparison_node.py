#!/usr/bin/env python

import rospy
import numpy as np
import message_filters
from people_tracking_v2.msg import HoCVectorArray, BodySizeArray, ComparisonScoresArray, ComparisonScores  # Import the custom message types
from std_msgs.msg import String
import os

class ComparisonNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('comparison_node', anonymous=True)
        
        # Synchronize the subscribers using message_filters
        hoc_sub = message_filters.Subscriber('/hoc_vectors', HoCVectorArray)
        pose_sub = message_filters.Subscriber('/pose_distances', BodySizeArray)
        
        ts = message_filters.ApproximateTimeSynchronizer([hoc_sub, pose_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.sync_callback)
        
        # Publisher for comparison scores
        self.comparison_pub = rospy.Publisher('/comparison/scores_array', ComparisonScoresArray, queue_size=10)
        
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
            rospy.loginfo(f"Loaded HoC data from {self.hoc_data_file}")
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
            rospy.loginfo(f"Loaded Pose data from {self.pose_data_file}")
        else:
            rospy.logerr(f"Pose data file {self.pose_data_file} not found")
            self.saved_pose_data = None

    def sync_callback(self, hoc_msg, pose_msg):
        """Callback function to handle synchronized HoC and pose data."""
        if self.saved_hue is None or self.saved_sat is None:
            rospy.logerr("No saved HoC data available for comparison")
            return

        # Log timestamps and IDs for synchronization verification
        rospy.loginfo(f"Synchronized messages: HoC timestamp: {hoc_msg.header.stamp}, Pose timestamp: {pose_msg.header.stamp}")

        comparison_scores_array = ComparisonScoresArray()
        comparison_scores_array.header.stamp = rospy.Time.now()

        for hoc_vector in hoc_msg.vectors:
            detection_id = hoc_vector.id
            hue_vector = hoc_vector.hue_vector
            sat_vector = hoc_vector.sat_vector
            hoc_distance_score = self.compute_hoc_distance_score(hue_vector, sat_vector)

            # Find corresponding pose data
            corresponding_pose = next((pose for pose in pose_msg.poses if pose.id == detection_id), None)
            if corresponding_pose:
                left_shoulder_hip_distance = corresponding_pose.left_shoulder_hip_distance
                right_shoulder_hip_distance = corresponding_pose.right_shoulder_hip_distance
                left_shoulder_hip_saved = np.mean(self.saved_pose_data['left_shoulder_hip_distance'])
                right_shoulder_hip_saved = np.mean(self.saved_pose_data['right_shoulder_hip_distance'])

                left_distance = self.compute_distance(left_shoulder_hip_distance, left_shoulder_hip_saved)
                right_distance = self.compute_distance(right_shoulder_hip_distance, right_shoulder_hip_saved)
                pose_distance_score = (left_distance + right_distance) / 2

                # Create ComparisonScores message
                comparison_scores_msg = ComparisonScores()
                comparison_scores_msg.header.stamp = rospy.Time.now()
                comparison_scores_msg.id = detection_id
                comparison_scores_msg.hoc_distance_score = hoc_distance_score
                comparison_scores_msg.pose_distance_score = pose_distance_score

                # Add to array
                comparison_scores_array.scores.append(comparison_scores_msg)

        # Publish comparison scores array
        self.comparison_pub.publish(comparison_scores_array)

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

if __name__ == '__main__':
    try:
        ComparisonNode()
    except rospy.ROSInterruptException:
        pass
