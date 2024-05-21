#!/usr/bin/env python

import rospy
import cv2
from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

class YoloNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO("yolov8l.pt")

        self.image_sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.image_callback)
        self.detection_pub = rospy.Publisher("/yolo_detections", String, queue_size=10)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Run the YOLOv8 model on the frame
        result = self.model(cv_image)[0]

        # Extract the detections from the result
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()

        rospy.loginfo(f"Detections: {len(labels)}")  # Log the number of detections using rospy.loginfo

        # Draw bounding boxes and labels on the frame
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)  # Set your desired color for bounding boxes
            thickness = 3
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                cv_image, f'{int(label)}: {score:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Display the frame (optional)
        # cv2.imshow("YOLOv8", cv_image)
        # cv2.waitKey(3)

        # Publish detection results
        detection_msg = String()
        detection_msg.data = f"Detections: {len(labels)}"
        self.detection_pub.publish(detection_msg)


def main():
    rospy.init_node('yolo_node', anonymous=True)
    yolo_node = YoloNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()