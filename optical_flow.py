import cv2
import numpy as np

def track_direction(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    direction = "Stationary"

    forward_movement = 0

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x = flow[...,0]
        forward_movement += np.mean(flow_x)
        prvs = next_frame

    cap.release()

    if forward_movement > 0.5:
        direction = "Forward"
    elif forward_movement < -0.5:
        direction = "Backward"

    return direction
