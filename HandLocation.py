import mediapipe as mp
import time


class HandLocation():

    def __init__(self):
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.location = mp.tasks.vision.HandLandmarker
        self.create_landmark()

    def create_landmark(self):
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        BaseOptions = mp.tasks.BaseOptions
        RunningMode = mp.tasks.vision.RunningMode
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            result_callback=update_result)

        self.location = self.location.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.location.detect_async(image=mp_image, timestamp_ms=int(time.time() * 1000))

    def close(self):
        self.location.close()



