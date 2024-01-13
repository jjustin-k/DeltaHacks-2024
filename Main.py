import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import os
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)
model = cap
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


process_frame = True

while True:
    ret, raw_frame = cap.read()
    focus_frame = np.ascontiguousarray(raw_frame)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=focus_frame)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=VisionRunningMode.VIDEO)
    with GestureRecognizer.create_from_options(options) as recognizer:
        gesture_recognition = recognizer.recognize(mp_image)
        cv2.imshow('Vid', raw_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()




