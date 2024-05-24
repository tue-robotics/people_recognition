#!/usr/bin/env python
import rospy
from people_tracking_v2.msg import Detection, DetectionArray
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
        
        # Initialize with some example data
        self.initialize_targets()
        
        rospy.spin()
    
    def initialize_targets(self):
        """Initialize the target memory with some example data."""
        initial_targets = [
            Target(nr_batch=0, time=rospy.Time.now(), colour_vector=[0.1] * 96),
            Target(nr_batch=1, time=rospy.Time.now(), colour_vector=[0.2] * 96),
            Target(nr_batch=2, time=rospy.Time.now(), colour_vector=[0.3] * 96)
        ]
        
        for target in initial_targets:
            self.targets[target.nr_batch] = target
        
        rospy.loginfo("Initialized target memory with example data.")
    
    def hoc_callback(self, msg):
        """Callback function to handle new HoC detections."""
        for detection in msg.detections:
            self.update_target(detection)
    
    def update_target(self, detection):
        """Update or add a target based on the detection."""
        batch_number = detection.nr_batch
        colour_vector = detection.colour_vector
        
        if batch_number in self.targets:
            # Update existing target
            existing_target = self.targets[batch_number]
            updated_target = Target(nr_batch=batch_number, time=existing_target.time, colour_vector=colour_vector)
            self.targets[batch_number] = updated_target
            rospy.loginfo(f"Updated target batch {batch_number}")
        else:
            # Add new target
            new_target = Target(nr_batch=batch_number, time=rospy.Time.now(), colour_vector=colour_vector)
            self.targets[batch_number] = new_target
            rospy.loginfo(f"Added new target batch {batch_number}")
        
        # Publish debug information
        debug_msg = String()
        debug_msg.data = f"Total targets: {len(self.targets)}"
        self.publisher_debug.publish(debug_msg)

if __name__ == '__main__':
    try:
        TargetMemoryNode()
    except rospy.ROSInterruptException:
        pass
