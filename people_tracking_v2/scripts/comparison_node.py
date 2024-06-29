#!/usr/bin/env python

import rospy
import numpy as np
import message_filters
from people_tracking_v2.msg import HoCVectorArray, BodySizeArray, ComparisonScoresArray, ComparisonScores  # Import the custom message types
from std_msgs.msg import String
import sys
import os

laptop = sys.argv[1]
name_subscriber_RGB = 'Webcam/image_raw' if laptop == "True" else '/hero/head_rgbd_sensor/rgb/image_raw'
depth_camera = False if sys.argv[2] == "False" else True
save_data = False if sys.argv[3] == "False" else True

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
        
        # Initialize variables for operator data
        self.init_phase = True
        self.init_duration = 10.0  # seconds
        self.start_time = rospy.get_time()
        self.hue_vectors = []
        self.sat_vectors = []
        self.val_vectors = []
        self.pose_data = []

        self.operator_hue_avg = None
        self.operator_sat_avg = None
        self.operator_val_avg = None
        self.operator_pose_median = None
        self.operator_data_set = False
        
        # Define the file paths for saving the data
        self.operator_npz_file_path = os.path.expanduser('~/hoc_data/operator_data.npz')
        self.detection_npz_file_path = os.path.expanduser('~/hoc_data/latest_detection_1_data.npz')
        
        rospy.spin()

    def sync_callback(self, hoc_array, pose_array):
        """Callback function to handle synchronized HoC and pose data."""
        rospy.loginfo("sync_callback invoked")

        current_time = rospy.get_time()
        if self.init_phase:
            # Accumulate data during initialization phase
            for hoc_msg, pose_msg in zip(hoc_array.vectors, pose_array.distances):
                if hoc_msg.id == 1:
                    self.hue_vectors.append(hoc_msg.hue_vector)
                    self.sat_vectors.append(hoc_msg.sat_vector)
                    self.val_vectors.append(hoc_msg.val_vector)
                    self.pose_data.append(pose_msg.head_feet_distance)
                    rospy.loginfo(f"Accumulating data for detection ID 1: HoC and Pose data")

            # Check if initialization phase is over
            if current_time - self.start_time > self.init_duration:
                self.init_phase = False
                # Calculate the median for pose data and normalized average for HoC data
                self.operator_pose_median = np.median(self.pose_data)
                self.operator_hue_avg = self.normalize_vector(np.mean(self.hue_vectors, axis=0))
                self.operator_sat_avg = self.normalize_vector(np.mean(self.sat_vectors, axis=0))
                self.operator_val_avg = self.normalize_vector(np.mean(self.val_vectors, axis=0))
                self.operator_data_set = True
                rospy.loginfo("Operator data initialized with median pose and normalized average HoC values.")
                
                # Save the accumulated operator data to file
                self.save_operator_data()
        else:
            if not self.operator_data_set:
                rospy.logerr("Operator data not yet set, waiting for initialization to complete")
                return

            if not hoc_array.vectors or not pose_array.distances:
                rospy.logerr("Received empty HoC or Pose array")
                return

            comparison_scores_array = ComparisonScoresArray()
            comparison_scores_array.header.stamp = hoc_array.header.stamp

            for hoc_msg, pose_msg in zip(hoc_array.vectors, pose_array.distances):
                rospy.loginfo(f"Processing Detection ID {hoc_msg.id}")

                comparison_scores_msg = ComparisonScores()
                comparison_scores_msg.header.stamp = hoc_msg.header.stamp
                comparison_scores_msg.header.frame_id = hoc_msg.header.frame_id
                comparison_scores_msg.id = hoc_msg.id

                hue_vector = np.array(hoc_msg.hue_vector)
                sat_vector = np.array(hoc_msg.sat_vector)
                val_vector = np.array(hoc_msg.val_vector)
                hoc_distance_score = self.compute_hoc_distance_score(hue_vector, sat_vector, val_vector)
                rospy.loginfo(f"Detection ID {hoc_msg.id}: HoC Distance score: {hoc_distance_score:.2f}")
                comparison_scores_msg.hoc_distance_score = hoc_distance_score

                head_feet_distance = pose_msg.head_feet_distance
                if head_feet_distance < 0:
                    rospy.loginfo(f"Skipping pose comparison for Detection ID {hoc_msg.id} due to negative pose distance")
                    comparison_scores_msg.pose_distance_score = -1
                else:
                    distance_score = self.compute_distance(head_feet_distance, self.operator_pose_median)
                    rospy.loginfo(f"Detection ID {pose_msg.id}: Pose Distance score: {distance_score:.2f}")
                    comparison_scores_msg.pose_distance_score = distance_score

                rospy.loginfo(f"Publishing scores - Detection ID {comparison_scores_msg.id}: HoC Distance score: {comparison_scores_msg.hoc_distance_score:.2f}, Pose Distance score: {comparison_scores_msg.pose_distance_score:.2f}")

                comparison_scores_array.scores.append(comparison_scores_msg)

                # Save the latest data for detection ID 1
                if hoc_msg.id == 1:
                    self.save_latest_detection_data(hue_vector, sat_vector, val_vector, head_feet_distance)

            self.comparison_pub.publish(comparison_scores_array)

    def normalize_vector(self, vector):
        """Normalize a vector to ensure the sum of elements is 1."""
        norm_vector = np.array(vector)
        norm_sum = np.sum(norm_vector)
        if norm_sum == 0:
            return norm_vector
        return norm_vector / norm_sum

    def compute_hoc_distance_score(self, hue_vector, sat_vector, val_vector):
        """Compute the Chi-Squared distance score between the current detection and saved data (HoC)."""
        hue_distance = self.compute_chi_squared_distance(hue_vector, self.operator_hue_avg)
        sat_distance = self.compute_chi_squared_distance(sat_vector, self.operator_sat_avg)
        val_distance = self.compute_chi_squared_distance(val_vector, self.operator_val_avg)
        
        return (hue_distance + sat_distance + val_distance)
    
    def compute_chi_squared_distance(self, vector1, vector2):
        """Compute the Chi-Squared distance between two vectors."""
        return 0.5 * np.sum(((vector1 - vector2) ** 2) / (vector1 + vector2 + 1e-10))  # Adding a small value to avoid division by zero

    def compute_distance(self, vector1, vector2):
        """Compute the Euclidean distance between two vectors (General)."""
        return np.linalg.norm(vector1 - vector2)

    #def save_operator_data(self):
        """Save the accumulated operator data to an .npz file."""
        np.savez(self.operator_npz_file_path, hue=self.operator_hue_avg, sat=self.operator_sat_avg, val=self.operator_val_avg, pose=self.operator_pose_median)
        rospy.loginfo(f"Saved accumulated operator data to {self.operator_npz_file_path}")

    #def save_latest_detection_data(self, hue_vector, sat_vector, val_vector, head_feet_distance):
        """Save the latest data for detection ID 1 to an .npz file."""
        np.savez(self.detection_npz_file_path, hue=hue_vector, sat=sat_vector, val=val_vector, pose=head_feet_distance)
        rospy.loginfo(f"Saved latest detection 1 data to {self.detection_npz_file_path}")

    #def publish_debug_info(self, hoc_distance_score, pose_distance_score, detection_id):
        """Publish debug information about the current comparison (General)."""
        debug_msg = String()
        debug_msg.data = f"Detection ID {detection_id}: HoC Distance score: {hoc_distance_score:.2f}, Pose Distance score: {pose_distance_score:.2f}"
        self.publisher_debug.publish(debug_msg)

if __name__ == '__main__':
    try:
        ComparisonNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
