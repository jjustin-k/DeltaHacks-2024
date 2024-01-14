import cv2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
import HandLocation
import mediapipe as mp
import numpy as np


def img_landmarks(rgb_image, detection_results: mp.tasks.vision.HandLandmarkerResult):
    try:
        if not detection_results.hand_landmarks:
            return rgb_image
        else:
            hand_landmarks_list = detection_results.hand_landmarks
            annotated_image = np.copy(rgb_image)

            for i in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[i]

                hand_landmarks_exp = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_exp.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks])

                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_exp,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

            return annotated_image
    except AttributeError:
        return rgb_image


def main():
    cap = cv2.VideoCapture(0)
    model = HandLocation.HandLocation()

    while True:
        ret, frame = cap.read()

        model.detect_async(frame)
        frame = img_landmarks(frame, model.result)

        cv2.imshow('Vid', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    model.close()
    cap.release()
    cv2.destroyAllWindows()

main()