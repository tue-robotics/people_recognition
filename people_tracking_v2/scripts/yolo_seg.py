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
        self.individual_segmented_image_pub = rospy.Publisher("/individual_segmented_images", Image, queue_size=10)
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
        detection_array.header.stamp = data.header.stamp  #  Use timestamp from incoming image

        # Create a copy of the image for bounding box visualization
        bounding_box_image = cv_image.copy()

        # Prepare the SegmentedImages message
        segmented_images_msg = SegmentedImages()
        segmented_images_msg.header.stamp = data.header.stamp  # Use the same timestamp as the YOLO image
        segmented_images_msg.ids = []  # Initialize the IDs list

        # Process each detection and create a Detection message, but only for humans (class 0)
        for i, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, masks)):
            detection = Detection()
            detection.id = i + 1  # Assign a unique ID to each detection
            detection.x1 = float(box[0])
            detection.y1 = float(box[1])
            detection.x2 = float(box[2])
            detection.y2 = float(box[3])
            detection.score = float(score)
            detection.label = int(label)
            detection_array.detections.append(detection)

            rospy.loginfo(f"Detection ID: {detection.id} for box: {box}")

            # Draw bounding boxes and labels on the bounding_box_image
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Initialize or update the Kalman Filter for this detection
            if detection.id not in self.kalman_filters:
                self.kalman_filters[detection.id] = KalmanFilterCV()
            kalman_filter = self.kalman_filters[detection.id]

            # Update the Kalman Filter with the new measurement
            kalman_filter.update(np.array([[x_center], [y_center]]))

            # Predict the next position
            kalman_filter.predict()
            x_pred, y_pred = kalman_filter.get_state()[:2]

            # Calculate the predicted bounding box based on the predicted center and original box size
            box_width = x2 - x1
            box_height = y2 - y1
            x_pred1, y_pred1 = int(x_pred - box_width / 2), int(y_pred - box_height / 2)
            x_pred2, y_pred2 = int(x_pred + box_width / 2), int(y_pred + box_height / 2)

            # Draw predicted bounding box
            cv2.rectangle(bounding_box_image, (x_pred1, y_pred1), (x_pred2, y_pred2), (255, 0, 0), 2)  # Blue box

            # Draw predicted position
            cv2.circle(bounding_box_image, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)

            # Calculate IoU
            iou = self.calculate_iou([x1, y1, x2, y2], [x_pred1, y_pred1, x_pred2, y_pred2])
            rospy.loginfo(f"Detection {detection.id}: IoU={iou:.2f}")

            color = (0, 255, 0)  # Set color for bounding boxes
            thickness = 3
            label_text = f'#{detection.id} {int(label)}: {score:.2f} IoU={iou:.2f}'
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
                segmented_images_msg.ids.append(detection.id)  # Append detection ID

                # Publish individual segmented images
                rospy.loginfo(f"Publishing individual segmented image with ID: {detection.id}")
                self.individual_segmented_image_pub.publish(segmented_image_msg)

        # Publish segmented images as a batch
        rospy.loginfo(f"Publishing Segmented Images with IDs: {segmented_images_msg.ids}")
        self.segmented_images_pub.publish(segmented_images_msg)

        # Publish bounding box image
        try:
            bounding_box_image_msg = self.bridge.cv2_to_imgmsg(bounding_box_image, "bgr8")
            bounding_box_image_msg.header.stamp = data.header.stamp  # Use the same timestamp
            rospy.loginfo("Publishing bounding box image")
            self.bounding_box_image_pub.publish(bounding_box_image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error while publishing: {e}")

        # Publish predicted detections
        self.detection_pub.publish(detection_array)

    def calculate_iou(self, box1, box2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

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
