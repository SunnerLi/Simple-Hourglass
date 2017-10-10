import numpy as np
import cv2

video_name = 'move.mp4'

cap = cv2.VideoCapture(video_name)
while cap.isOpened():
    is_cap_open, frame = cap.read()
    fixed_img = cv2.resize(frame, (780, 1040))
    cv2.imshow('res', fixed_img)
    cv2.waitKey(100)