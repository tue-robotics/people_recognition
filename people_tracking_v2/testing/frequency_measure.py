#!/usr/bin/env python

import rospy
from people_tracking_v2.msg import HoCVectorArray, BodySizeArray, DetectionArray, DecisionResult  # Replace with actual package name
import numpy as np
import csv
import os
import threading

class FrequencyMonitor:
    def __init__(self, topic_name, msg_type):
        self.topic_name = topic_name
        self.times = []
        self.subscriber = rospy.Subscriber(topic_name, msg_type, self.callback)

    def callback(self, msg):
        current_time = rospy.get_time()
        self.times.append(current_time)

    def compute_frequencies(self):
        if len(self.times) < 2:
            rospy.loginfo(f"Not enough data to compute frequencies for {self.topic_name}.")
            return None, None

        time_intervals = np.diff(self.times)
        mean_interval = np.mean(time_intervals)
        max_interval = np.max(time_intervals)

        mean_frequency = 1.0 / mean_interval if mean_interval > 0 else 0
        worst_case_frequency = 1.0 / max_interval if max_interval > 0 else 0

        return mean_frequency, worst_case_frequency

def save_frequencies(monitors, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['topic', 'mean_frequency', 'worst_case_frequency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for monitor in monitors:
            mean_frequency, worst_case_frequency = monitor.compute_frequencies()
            if mean_frequency is not None and worst_case_frequency is not None:
                writer.writerow({
                    'topic': monitor.topic_name,
                    'mean_frequency': mean_frequency,
                    'worst_case_frequency': worst_case_frequency
                })

    rospy.loginfo(f"Frequency data saved to {output_file}")

def timer_callback(monitors, output_file):
    save_frequencies(monitors, output_file)
    rospy.signal_shutdown("Data collection complete.")

def main():
    rospy.init_node('frequency_monitor', anonymous=True)

    # Define the topics and their corresponding message types
    topics = [
        ('/hoc_vectors', HoCVectorArray),
        ('/pose_distances', BodySizeArray),
        ('/detections_info', DetectionArray),
        ('/decision/result', DecisionResult)
    ]

    monitors = [FrequencyMonitor(topic, msg_type) for topic, msg_type in topics]

    # Set up a timer to save data after 1 minute (60 seconds)
    output_file = '/home/miguel/Documents/BEP-Testing/Test Case 1/Frequency Measurement/frequency_measurement.csv'
    timer = threading.Timer(60, timer_callback, [monitors, output_file])
    timer.start()

    rospy.spin()

if __name__ == '__main__':
    main()
