from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO('yolov8n-pose.pt')
video_path = 0
cap = cv2.VideoCapture(video_path)
# 这里若cap = cv2.VideoCapture(0)
# 便是打开电脑的默认摄像头
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
