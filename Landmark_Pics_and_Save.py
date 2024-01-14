import os
import cv2
import mediapipe as mp


def list_subfolders(p):
    items = os.listdir(p)
    subf = [
        item for item in items if os.path.isdir(
            os.path.join(p, item))]
    return subf


def get_num_files(p):
    count = 0
    for path in os.listdir(p):
        if os.path.isfile(os.path.join(p, path)):
            count += 1
    return count


def process_image(p):
    image = cv2.imread(p)

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
    cv2.imwrite(filename, image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


resource_path = "resources/SigNN Character Database/"
subfolders = list_subfolders(resource_path)
for subfolder in subfolders:
    subfolder_path = os.path.join(resource_path, subfolder)
    num_files = get_num_files(subfolder_path)
    i = 1
    while i <= num_files:
        image_path = subfolder_path + "/" + str(i) + ".jpg"
        process_image(image_path)
        i += 1


# # Load an image using OpenCV
# image_path = "resources/SigNN Character Database/A\1.png"
# image = cv2.imread(image_path)
#
# # Convert the BGR image to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Create a MediaPipe hands instance
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# # Create a drawing module instance
# mp_drawing = mp.solutions.drawing_utils
#
# # Process the image
# results = hands.process(image_rgb)
#
# # If you want to get the annotated image with landmarks
# if results.multi_hand_landmarks:
#     for hand_landmarks in results.multi_hand_landmarks:
#         # Draw landmarks on the image
#         mp_drawing.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
# directory = r'C:\Deltahacks_Photos'
# filename = 'savedImage.jpg'
# os.chdir(directory)
#
# # Display the image with landmarks
# cv2.imshow('MediaPipe Hands', image_rgb)
# cv2.imwrite(filename,image_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()