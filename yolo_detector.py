import torch
import cv2

def detect_objects(video_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    cap = cv2.VideoCapture(video_path)

    skater_detected = False
    skateboard_detected = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        labels = results.xyxyn[0][:, -1].numpy()

        if 0 in labels:  # Assuming label '0' is person
            skater_detected = True
        if 36 in labels:  # Assuming label '36' is skateboard
            skateboard_detected = True

        if skater_detected and skateboard_detected:
            break

    cap.release()
    return skater_detected, skateboard_detected
