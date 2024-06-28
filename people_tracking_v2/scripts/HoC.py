#!/usr/bin/env python

import rospy
import message_filters
from people_tracking_v2.msg import SegmentedImages, HoCVectorArray, HoCVector, DetectionArray  # Custom messages
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32  # Import the Float32 message type
import cv2
import numpy as np

class HoCNode:
    def __init__(self, initialize_node=True):
        if initialize_node:
            rospy.init_node('hoc_node', anonymous=True)
        
        self.bridge = CvBridge()

        # Synchronize segmented_images and detections_info messages
        segmented_images_sub = message_filters.Subscriber('/segmented_images', SegmentedImages)
        detections_sub = message_filters.Subscriber('/detections_info', DetectionArray)

        ts = message_filters.ApproximateTimeSynchronizer([segmented_images_sub, detections_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.sync_callback)

        # Publisher for HoC vectors
        self.hoc_vector_pub = rospy.Publisher('/hoc_vectors', HoCVectorArray, queue_size=10)

        if initialize_node:
            rospy.spin()

    def sync_callback(self, segmented_images_msg, detections_msg):
        """Callback function to handle synchronized segmented images and detections info."""
        if not detections_msg.detections:
            rospy.loginfo("No detections available, skipping processing.")
            return
        
        hoc_vectors = HoCVectorArray()
        hoc_vectors.header.stamp = segmented_images_msg.header.stamp  # Use the same timestamp as the incoming message
        
        for i, segmented_image_msg in enumerate(segmented_images_msg.images):
            try:
                segmented_image = self.bridge.imgmsg_to_cv2(segmented_image_msg, "bgr8")
                hoc_hue, hoc_sat, hoc_val = self.compute_hoc(segmented_image)

                # Extract the ID from the incoming message
                detection_id = segmented_images_msg.ids[i]

                # Find the corresponding detection with the same ID
                detection = next((d for d in detections_msg.detections if d.id == detection_id), None)
                if detection is None:
                    rospy.logerr(f"No matching detection found for ID: {detection_id}")
                    continue

                # Create HoCVector message
                hoc_vector = HoCVector()
                hoc_vector.id = detection_id
                hoc_vector.hue_vector = self.normalize_vector(hoc_hue).tolist()
                hoc_vector.sat_vector = self.normalize_vector(hoc_sat).tolist()
                hoc_vector.val_vector = self.normalize_vector(hoc_val).tolist()
                hoc_vectors.vectors.append(hoc_vector)

            except CvBridgeError as e:
                rospy.logerr(f"Failed to convert segmented image: {e}") 
            except IndexError as e:
                rospy.logerr(f"IndexError: {e}. This might happen if there are more segmented images than detections.")
        
        # Publish the HoC vectors as a batch
        self.hoc_vector_pub.publish(hoc_vectors)
        
    def compute_hoc(self, segmented_image):
        # Convert to HSV
        hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
        
        # Create a mask to ignore black pixels
        mask = cv2.inRange(hsv, (0, 0, 1), (180, 255, 255))
        
        # Use the same number of bins for Hue, Saturation, and Value
        bins = 256
        
        # Compute histogram for Hue, Saturation, and Value using the mask
        hist_hue = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
        hist_sat = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])
        hist_val = cv2.calcHist([hsv], [2], mask, [bins], [0, 256])
        
        # Normalize histograms
        hist_hue = self.normalize_vector(hist_hue)
        hist_sat = self.normalize_vector(hist_sat)
        hist_val = self.normalize_vector(hist_val)
        
        # Flatten the histograms
        return hist_hue.flatten(), hist_sat.flatten(), hist_val.flatten()

    def normalize_vector(self, vector):
        """Normalize a vector to ensure the sum of elements is 1."""
        norm_vector = np.array(vector)
        norm_sum = np.sum(norm_vector)
        if norm_sum == 0:
            return norm_vector
        return norm_vector / norm_sum

if __name__ == '__main__':
    try:
        HoCNode()
    except rospy.ROSInterruptException:
        pass
