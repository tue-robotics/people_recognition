#!/usr/bin/env python
import rospy
import csv
import numpy as np
from people_tracking_v2.msg import DetectionArray
from std_msgs.msg import String
from collections import namedtuple

# Named tuple for storing target information
Target = namedtuple("Target", ["nr_batch", "time", "colour_vector"])

class TargetMemoryNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('target_memory', anonymous=True)
        
        # Subscriber to HoC detections
        self.subscriber_hoc = rospy.Subscriber('/hero/HoC', DetectionArray, self.hoc_callback)
        
        # Publisher for debug information or status updates
        self.publisher_debug = rospy.Publisher('/hero/target_memory/debug', String, queue_size=10)
        
        # Dictionary to store targets with their batch number as key
        self.targets = {}
        
        # Load saved HoC data
        self.load_hoc_data()
        
        rospy.spin()
    
    def load_hoc_data(self):
        """Load the saved HoC data from the CSV file."""
        try:
            with open('hoc_data.csv', 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    nr_batch = int(row[0])
                    time = rospy.Time(float(row[1]))
                    colour_vector = list(map(float, row[2:]))
                    self.targets[nr_batch] = Target(nr_batch=nr_batch, time=time, colour_vector=colour_vector)
            rospy.loginfo("Loaded HoC data from hoc_data.csv")
        except Exception as e:
            rospy.logerr(f"Failed to load HoC data: {e}")

    def hoc_callback(self, msg):
        """Callback function to handle new HoC detections."""
        for detection in msg.detections:
            similarity_score = self.compute_similarity(detection)
            self.update_target(detection, similarity_score)
    
    def update_target(self, detection, similarity_score):
        """Update or add a target based on the detection."""
        batch_number = detection.nr_batch
        colour_vector = detection.colour_vector
        
        if batch_number in self.targets:
            # Update existing target
            existing_target = self.targets[batch_number]
            updated_target = Target(nr_batch=batch_number, time=existing_target.time, colour_vector=colour_vector)
            self.targets[batch_number] = updated_target
            rospy.loginfo(f"Updated target batch {batch_number} with similarity score {similarity_score:.2f}")
        else:
            # Add new target
            new_target = Target(nr_batch=batch_number, time=rospy.Time.now(), colour_vector=colour_vector)
            self.targets[batch_number] = new_target
            rospy.loginfo(f"Added new target batch {batch_number} with similarity score {similarity_score:.2f}")
        
        # Publish debug information
        debug_msg = String()
        debug_msg.data = f"Total targets: {len(self.targets)} | Last similarity score: {similarity_score:.2f}"
        self.publisher_debug.publish(debug_msg)

    def compute_similarity(self, detection):
        """Compute the similarity score between the current detection and saved targets."""
        max_similarity = 0
        current_vector = np.array(detection.colour_vector)
        
        for target in self.targets.values():
            saved_vector = np.array(target.colour_vector)
            similarity = np.dot(current_vector, saved_vector) / (np.linalg.norm(current_vector) * np.linalg.norm(saved_vector))
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity

if __name__ == '__main__':
    try:
        TargetMemoryNode()
    except rospy.ROSInterruptException:
        pass
