#!/usr/bin/env python

import cv2
from ultralytics import YOLO
import supervision as sv
print("OpenCV version:", cv2.__version__)

def main():
    cap = cv2.VideoCapture(0)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()
        cv2.imshow("yolov8", frame)

        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)

        frame = box_annotator.annotate(scene=frame, detections=detections)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()