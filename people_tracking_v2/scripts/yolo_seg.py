#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image
from people_tracking_v2.msg import Detection, DetectionArray, SegmentedImages
from cv_bridge import CvBridge, CvBridgeError

# Add the path to the `kalman_filter.py` module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'people_tracking'))
from kalman_filter import KalmanFilterCV  # Import the Kalman Filter class

class YoloSegNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n-seg.pt")  # Ensure the model supports segmentation

        self.image_sub = rospy.Subscriber("/Webcam/image_raw", Image, self.image_callback)
        self.segmented_images_pub = rospy.Publisher("/segmented_images", SegmentedImages, queue_size=10)
        self.bounding_box_image_pub = rospy.Publisher("/bounding_box_image", Image, queue_size=10)
        self.detection_pub = rospy.Publisher("/hero/predicted_detections", DetectionArray, queue_size=10)

        # Initialize the Kalman Filter
        self.kalman_filters = {}

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        # Run the YOLOv8 model on the frame
        results = self.model(cv_image)[0]

        # Extract the detections from the result
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy()
        masks = results.masks.data.cpu().numpy() if results.masks else None

        rospy.loginfo(f"Total Detections: {len(labels)}")  # Log the total number of detections

        detection_array = DetectionArray()

        # Create a copy of the image for bounding box visualization
        bounding_box_image = cv_image.copy()

        # Prepare the SegmentedImages message
        segmented_images_msg = SegmentedImages()
        segmented_images_msg.header.stamp = rospy.Time.now()

        # Process each detection and create a Detection message, but only for humans (class 0)
        human_detections = [(box, score, label, mask) for box, score, label, mask in zip(boxes, scores, labels, masks) if int(label) == 0]
        rospy.loginfo(f"Human Detections: {len(human_detections)}")  # Log the number of human detections

        for i, (box, score, label, mask) in enumerate(human_detections):
            detection = Detection()
            detection.x1 = float(box[0])
            detection.y1 = float(box[1])
            detection.x2 = float(box[2])
            detection.y2 = float(box[3])
            detection.score = float(score)
            detection.label = int(label)
            detection_array.detections.append(detection)

            # Draw bounding boxes and labels on the bounding_box_image
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Initialize or update the Kalman Filter for this detection
            if i not in self.kalman_filters:
                self.kalman_filters[i] = KalmanFilterCV()
            kalman_filter = self.kalman_filters[i]

            # Update the Kalman Filter with the new measurement
            kalman_filter.update(np.array([[x_center], [y_center]]))

            # Predict the next position
            kalman_filter.predict()
            x_pred, y_pred = kalman_filter.get_state()[:2]

            # Draw predicted position
            cv2.circle(bounding_box_image, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)

            color = (0, 255, 0)  # Set color for bounding boxes
            thickness = 3
            label_text = f'#{i+1} {int(label)}: {score:.2f}'
            cv2.rectangle(bounding_box_image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                bounding_box_image, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

            # Apply segmentation mask to the cv_image
            if mask is not None:
                # Resize mask to match the original image dimensions
                resized_mask = cv2.resize(mask, (cv_image.shape[1], cv_image.shape[0]))
                resized_mask = resized_mask.astype(np.uint8)

                # Apply mask to the original image
                segmented_image = cv2.bitwise_and(cv_image, cv_image, mask=resized_mask)
                segmented_image_msg = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
                segmented_images_msg.images.append(segmented_image_msg)

        # Publish segmented images as a batch
            self.segmented_images_pub.publish(segmented_images_msg)

        # Publish bounding box image
        try:
            bounding_box_image_msg = self.bridge.cv2_to_imgmsg(bounding_box_image, "bgr8")
            rospy.loginfo("Publishing bounding box image")
            self.bounding_box_image_pub.publish(bounding_box_image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error while publishing: {e}")

        # Publish predicted detections
        self.detection_pub.publish(detection_array)

def main():
    rospy.init_node('yolo_seg_node', anonymous=True)
    yolo_node = YoloSegNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
