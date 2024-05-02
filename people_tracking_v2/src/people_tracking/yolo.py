#!/usr/bin/env python

import cv2
from ultralytics import YOLO

def main():
    cap = cv2.VideoCapture(0)

    model = YOLO("yolov8l.pt")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run the YOLOv8 model on the frame
        result = model(frame)[0]
        
        # Print the result to understand its structure
        print(result)

        # Extract the detections from the result
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()

        # Draw bounding boxes and labels on the frame
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)  # Set your desired color for bounding boxes
            thickness = 3
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            frame = cv2.putText(
                frame, f'{int(label)}: {score:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Display the frame
        cv2.imshow("YOLOv8", frame)

        # Exit the loop if 'Esc' key is pressed
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
