import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult


class HandLandmarker:
    def __init__(self):
        self.base_options = python.BaseOptions(
            model_asset_path='bundles/hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options,
                                                    num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def detect_landmark_from_file(self, image_path: str) -> HandLandmarkerResult:
        image = mp.Image.create_from_file(image_path)
        return self.detector.detect(image)

    def detect_landmark_from_data(self, data) -> HandLandmarkerResult:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=data)
        return self.detector.detect(image)
