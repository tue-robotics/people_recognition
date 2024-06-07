#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image
from people_tracking_v2.msg import Detection, DetectionArray, SegmentedImages
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32, Int32  # Import the Int32 message type for operator ID

import sys
import os
import rospkg
import time

# Add the path to the `kalman_filter.py` module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'people_tracking'))
from kalman_filter import KalmanFilterCV  # Import the Kalman Filter class


laptop = sys.argv[1]
name_subscriber_RGB = 'Webcam/image_raw' if laptop == "True" else '/hero/head_rgbd_sensor/rgb/image_raw'
depth_camera = False if sys.argv[2] == "False" else True
save_data = False if sys.argv[3] == "False" else True

class YoloSegNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n-seg.pt")  # Ensure the model supports segmentation

        self.image_sub = rospy.Subscriber(name_subscriber_RGB, Image, self.image_callback)
        if depth_camera:
            self.depth_sub = rospy.Subscriber('/hero/head_rgbd_sensor/depth_registered/image_raw', Image, self.depth_image_callback)

        self.segmented_images_pub = rospy.Publisher("/segmented_images", SegmentedImages, queue_size=10)
        self.individual_segmented_image_pub = rospy.Publisher("/individual_segmented_images", Image, queue_size=10)
        self.bounding_box_image_pub = rospy.Publisher("/bounding_box_image", Image, queue_size=10)
        self.detection_pub = rospy.Publisher("/detections_info", DetectionArray, queue_size=10)

        # Subscriber for operator ID
        self.operator_id_sub = rospy.Subscriber('/decision/result', Int32, self.operator_id_callback)

        # Initialize the Kalman Filter for the operator
        self.kalman_filter_operator = KalmanFilterCV()
        self.operator_id = None  # To be set by an external node

        self.iou_threshold_pub = rospy.Publisher('/iou_threshold', Float32, queue_size=10)
        self.iou_threshold = 0.9  # Default threshold value

        # Initialize variables for saving data and depth processing
        self.latest_image = None
        self.depth_images = []
        self.batch_nr = 0

        if save_data:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path("people_tracking_v2")
            current_time = time.strftime("%Y%m%d-%H%M%S")
            self.save_path = os.path.join(package_path, f'data/{current_time}_test/')
            os.makedirs(self.save_path, exist_ok=True)
            rospy.loginfo(f"Data will be saved to: {self.save_path}")

    def operator_id_callback(self, msg):
        """Callback function to update the operator ID."""
        self.operator_id = msg.data
        rospy.loginfo(f"Updated operator ID to: {self.operator_id}")

    def set_operator(self, operator_id):
        """Set the ID of the operator to track."""
        self.operator_id = operator_id

    def depth_image_callback(self, data):
        """Store the latest depth image. Only the most recent depth images are stored."""
        while len(self.depth_images) > 50:
            self.depth_images.pop(0)
        self.depth_images.append(data)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.batch_nr = data.header.seq
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        # Import Depth Image
        if depth_camera:
            cv_depth_image = self.bridge.imgmsg_to_cv2(self.depth_images[-1], desired_encoding='passthrough')
        else:
            cv_depth_image = None

        # Save RGB and Depth Images if required
        if save_data:
            cv2.imwrite(f"{self.save_path}{self.batch_nr}.png", cv_image)
            if depth_camera:
                cv2.imwrite(f"{self.save_path}{self.batch_nr}_depth.png", cv_depth_image)

        # Run the YOLOv8 model on the frame
        results = self.model(cv_image)[0]

        # Extract the detections from the result
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy()
        masks = results.masks.data.cpu().numpy() if results.masks else None

        rospy.loginfo(f"Total Detections: {len(labels)}")  # Log the total number of detections

        detection_array = DetectionArray()
        detection_array.header.stamp = data.header.stamp  # Use timestamp from incoming image

        # Create a copy of the image for bounding box visualization
        bounding_box_image = cv_image.copy()

        # Prepare the SegmentedImages message
        segmented_images_msg = SegmentedImages()
        segmented_images_msg.header.stamp = data.header.stamp  # Use the same timestamp as the YOLO image
        segmented_images_msg.ids = []  # Initialize the IDs list

        operator_box = None

        # Process each detection and create a Detection message
        for i, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, masks)):
            if int(label) != 0:  # Only process humans (class 0)
                continue

            detection = Detection()
            detection.id = i + 1  # Assign a unique ID to each detection
            detection.x1 = float(box[0])
            detection.y1 = float(box[1])
            detection.x2 = float(box[2])
            detection.y2 = float(box[3])
            detection.score = float(score)
            detection.label = int(label)

            # Extract depth value
            if depth_camera and cv_depth_image is not None:
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                depth_value = cv_depth_image[y_center, x_center]
                detection.depth = float(depth_value)
            else:
                detection.depth = -1.0  # Use -1.0 as a placeholder if depth is not available

            detection_array.detections.append(detection)

            rospy.loginfo(f"Detection ID: {detection.id} for box: {box}, depth: {detection.depth}")

            # Draw bounding boxes and labels on the bounding_box_image
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)  # Set color for bounding boxes
            thickness = 3
            label_text = f'#{detection.id} {int(label)}: {score:.2f}'

            cv2.rectangle(bounding_box_image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                bounding_box_image, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

            if self.operator_id is not None and detection.id == self.operator_id:
                operator_box = [x1, y1, x2, y2]
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                self.kalman_filter_operator.update(np.array([[x_center], [y_center]]))

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

        # Predict the next position of the operator
        if operator_box is not None:
            self.kalman_filter_operator.predict()
            x_pred, y_pred = self.kalman_filter_operator.get_state()[:2]

            box_width = operator_box[2] - operator_box[0]
            box_height = operator_box[3] - operator_box[1]
            x_pred1, y_pred1 = int(x_pred - box_width / 2), int(y_pred - box_height / 2)
            x_pred2, y_pred2 = int(x_pred + box_width / 2), int(y_pred + box_height / 2)

            # Draw predicted bounding box
            cv2.rectangle(bounding_box_image, (x_pred1, y_pred1), (x_pred2, y_pred2), (255, 0, 0), 2)  # Blue box

            # Draw predicted position
            cv2.circle(bounding_box_image, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)

            # Calculate IoU for all detections with the operator's predicted bounding box
            for detection in detection_array.detections:
                x1, y1, x2, y2 = int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)
                iou = self.calculate_iou([x1, y1, x2, y2], [x_pred1, y_pred1, x_pred2, y_pred2])
                detection.iou = iou  # Set the IoU value
                rospy.loginfo(f"Detection {detection.id}: IoU with operator={iou:.2f}")

                # Update the bounding box image with IoU values
                label_text = f'#{detection.id} {int(detection.label)}: {detection.score:.2f} IoU={iou:.2f}'
                cv2.putText(
                    bounding_box_image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

        # Publish the IoU threshold
        iou_threshold_msg = Float32()
        iou_threshold_msg.data = self.iou_threshold
        self.iou_threshold_pub.publish(iou_threshold_msg)

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
        print(f"Use Depth: {depth_camera}, Camera Source: {name_subscriber_RGB}")
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
