import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import os
import math


detector = HandDetector(maxHands=2, detectionCon=0.8)


def display():
    cap = cv2.VideoCapture(0)
    process_frame = True
    while True:
        ret, raw_frame = cap.read()
        focus_frame = np.ascontiguousarray(raw_frame)
        cv2.imshow('Vid', raw_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return


display()



