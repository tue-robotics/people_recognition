#!/usr/bin/env python

import rospy
import numpy as np
from people_tracking_v2.msg import HoCVector, BodySize  # Import the custom message types
from std_msgs.msg import String
import os

class ComparisonNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('comparison_node', anonymous=True)
        
        # Subscribers to HoC vectors and BodySize
        self.subscriber_hoc = rospy.Subscriber('/hoc_vectors', HoCVector, self.hoc_callback)
        self.subscriber_pose = rospy.Subscriber('/pose_distances', BodySize, self.pose_callback)
        
        # Publisher for debug information or status updates
        self.publisher_debug = rospy.Publisher('/comparison/debug', String, queue_size=10)
        
        # Load saved HoC and Pose data
        self.hoc_data_file = os.path.expanduser('~/hoc_data/hoc_data.npz')
        self.pose_data_file = os.path.expanduser('~/pose_data/pose_data.npz')
        self.load_hoc_data()
        self.load_pose_data()

        # Initialize storage for the latest incoming data
        self.latest_hoc_vectors = {}
        self.latest_pose_data = {}

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

    def hoc_callback(self, msg):
        """Callback function to handle new HoC detections (HoC)."""
        if self.saved_hue is None or self.saved_sat is None:
            rospy.logerr("No saved HoC data available for comparison")
            return
        
        # Store the HoC vector with its ID
        self.latest_hoc_vectors[msg.id] = msg
        self.compare_data()

    def pose_callback(self, msg):
        """Callback function to handle new BodySize data (Pose)."""
        # Store the pose data with its ID
        self.latest_pose_data[msg.id] = msg
        self.compare_data()

    def compare_data(self):
        """Compare HoC and pose data if both are available (General)."""
        if not self.latest_hoc_vectors or not self.latest_pose_data or self.saved_pose_data is None:
            return

        # Iterate through the detections by ID
        for detection_id, hoc_vector in self.latest_hoc_vectors.items():
            if detection_id not in self.latest_pose_data:
                continue

            # Compare HoC data
            hue_vector = hoc_vector.hue_vector
            sat_vector = hoc_vector.sat_vector
            hoc_distance_score = self.compute_hoc_distance_score(hue_vector, sat_vector)
            rospy.loginfo(f"Detection ID {detection_id}: HoC Distance score: {hoc_distance_score:.2f}")

            # Compare pose data
            pose_data = self.latest_pose_data[detection_id]
            left_shoulder_hip_distance = pose_data.left_shoulder_hip_distance
            right_shoulder_hip_distance = pose_data.right_shoulder_hip_distance
            left_shoulder_hip_saved = np.mean(self.saved_pose_data['left_shoulder_hip_distance'])
            right_shoulder_hip_saved = np.mean(self.saved_pose_data['right_shoulder_hip_distance'])

            left_distance = self.compute_distance(left_shoulder_hip_distance, left_shoulder_hip_saved)
            right_distance = self.compute_distance(right_shoulder_hip_distance, right_shoulder_hip_saved)
            pose_distance_score = (left_distance + right_distance) / 2
            rospy.loginfo(f"Detection ID {detection_id}: Pose Distance score: {pose_distance_score:.2f}")

            # Publish debug information
            self.publish_debug_info(hoc_distance_score, pose_distance_score, detection_id)

        # Clear the latest data after processing
        self.latest_hoc_vectors.clear()
        self.latest_pose_data.clear()

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
