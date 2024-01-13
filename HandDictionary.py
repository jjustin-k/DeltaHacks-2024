import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import os
import math
import os
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def process_photos(folder_path):
    # List all files in the specified folder
    files = os.listdir(folder_path)

    # Iterate through each file in the folder
    for file in files:
        # Check if the file is an image (you can customize this check based on your file extensions)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, file)

            # Open the image using Pillow
            image = Image.open(image_path)

            # Process the image (you can add your own image processing logic here)

            # Example: Display the image size
            print(f"Image: {file}, Size: {image.size}")

            # Close the image file
            image.close()

# Replace 'your_folder_path' with the path to your folder containing photos
folder_path = 'your_folder_path'
process_photos(folder_path)



