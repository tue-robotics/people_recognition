#!/usr/bin/env python

import rospy
import numpy as np
from people_tracking_v2.msg import BodySizeArray  # Updated to match the data type published on /pose_distances
import os

class SavePoseDataNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('save_pose_data_node', anonymous=True)
        
        # Subscriber to pose distance data
        self.pose_sub = rospy.Subscriber('/pose_distances', BodySizeArray, self.pose_callback)
        
        # File to save pose data
        self.pose_data_file = os.path.expanduser('~/pose_data/pose_data.npz')
        
        rospy.spin()
        
    def pose_callback(self, msg):
        if not msg.distances:
            rospy.loginfo("No pose distances received, skipping saving.")
            return

        rospy.loginfo(f"Received {len(msg.distances)} pose distances")
        
        # Store only the first pose distance for simplicity
        pose_distance = msg.distances[0]
        head_feet_distance = np.array([pose_distance.head_feet_distance])

        # Save the first pose distance in a single .npz file
        np.savez(self.pose_data_file, head_feet_distance=head_feet_distance)
        rospy.loginfo(f'Saved pose data to {self.pose_data_file}')

if __name__ == '__main__':
    try:
        SavePoseDataNode()
    except rospy.ROSInterruptException:
        pass
