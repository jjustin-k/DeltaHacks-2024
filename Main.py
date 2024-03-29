import cv2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
import HandLocation
import mediapipe as mp
import numpy as np
import spacy


def recognize_words(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    words = [token.text for token in doc]
    string = ""
    for i in words:
        string += i
    return string


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

    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="exported_model/gesture_recognizer.task"),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        )

    recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)
    model = HandLocation.HandLocation()
    sentence = " "
    index = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        model.detect_async(frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        gest_result = recognizer.recognize(mp_image).gestures
        for gesture in gest_result:
            gest_type = [category.category_name for category in gesture]
            if sentence[index] != str(gest_type[0]):
                sentence += str(gest_type[0])
                index = len(sentence)-1
            cv2.putText(frame, text=gest_type[0], org=(50,50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=(0, 255, 155))

        frame = img_landmarks(frame, model.result)

        cv2.imshow('vid', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    model.close()
    cap.release()
    cv2.destroyAllWindows()
    print(recognize_words(sentence))


main()
