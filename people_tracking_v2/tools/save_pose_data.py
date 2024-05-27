#!/usr/bin/env python

import rospy
import numpy as np
from people_tracking_v2.msg import PoseDistance
import os

class SavePoseDataNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('save_pose_data_node', anonymous=True)
        
        # Subscriber to pose distance data
        self.pose_sub = rospy.Subscriber('/pose_distances', PoseDistance, self.pose_callback)
        
        # File to save pose data
        self.pose_data_file = os.path.expanduser('~/pose_data/pose_data.npz')
        
        # Ensure the directory exists
        pose_data_dir = os.path.dirname(self.pose_data_file)
        if not os.path.exists(pose_data_dir):
            os.makedirs(pose_data_dir)
            rospy.loginfo(f"Created directory: {pose_data_dir}")
        else:
            rospy.loginfo(f"Using existing directory: {pose_data_dir}")
        
        self.pose_data = {
            'left_shoulder_hip_distance': [],
            'right_shoulder_hip_distance': []
        }
        
        rospy.spin()
        
    def pose_callback(self, msg):
        rospy.loginfo("Received pose distance data")
        
        # Append the distances to the data dictionary
        self.pose_data['left_shoulder_hip_distance'].append(msg.left_shoulder_hip_distance)
        self.pose_data['right_shoulder_hip_distance'].append(msg.right_shoulder_hip_distance)
        rospy.loginfo(f"Left Shoulder-Hip Distance: {msg.left_shoulder_hip_distance:.2f}")
        rospy.loginfo(f"Right Shoulder-Hip Distance: {msg.right_shoulder_hip_distance:.2f}")
        
        # Save the pose data periodically or based on some condition
        if len(self.pose_data['left_shoulder_hip_distance']) >= 10:  # Save every 10 messages as an example
            self.save_pose_data()
        
    def save_pose_data(self):
        """Save the collected pose data to a .npz file."""
        np.savez(self.pose_data_file, **self.pose_data)
        rospy.loginfo(f"Saved pose data to {self.pose_data_file}")
        self.pose_data = {
            'left_shoulder_hip_distance': [],
            'right_shoulder_hip_distance': []
        }  # Clear the data after saving

if __name__ == '__main__':
    try:
        SavePoseDataNode()
    except rospy.ROSInterruptException:
        pass
