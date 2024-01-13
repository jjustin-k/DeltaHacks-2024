import os
from os import listdir

import cv2
import mediapipe as mp

# Load an image using OpenCV
image_path = 'C:\WIN_20240113_17_30_27_Pro.jpg'
image = cv2.imread(image_path)

# Convert the BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a MediaPipe hands instance
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Create a drawing module instance
mp_drawing = mp.solutions.drawing_utils

# Process the image
results = hands.process(image_rgb)

# If you want to get the annotated image with landmarks
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw landmarks on the image
        mp_drawing.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

directory = r'C:\Deltahacks_Photos'
filename = 'savedImage.jpg'
os.chdir(directory)

# Display the image with landmarks
cv2.imshow('MediaPipe Hands', image_rgb)
cv2.imwrite(filename,image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()