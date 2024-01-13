import cv2
from cvzone.HandTrackingModule import HandDetector
import HandLocation
import mediapipe as mp
import numpy as np

import os
import math

cap = cv2.VideoCapture(0)

process_frame = True

while True:
    ret, raw_frame = cap.read()
    focus_frame = np.ascontiguousarray(raw_frame)

    model = HandLocation.HandLocation()
    model.detect_async(raw_frame)

    print(model.result)
    cv2.imshow('Vid', raw_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




