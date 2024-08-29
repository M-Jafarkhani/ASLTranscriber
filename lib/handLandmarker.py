import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult


class HandLandmarker:
    """
    A Python class for extarcting hand landmarks from an image

    ...

    Attributes
    ----------
    base_options : BaseOptions
        MediaPipe's BaseOptions which are loaed from 'bundles/hand_landmarker.task' pre-trained model.

    options : HandLandmarkerOptions
        Options for initializing hand detetctor.

    detector : HandLandmarker
        HandLandmarker object to detect hand and extract landmarks and handedness, if a hand is found.

    Methods
    -------
    detect_landmark_from_file(image_path: str) -> HandLandmarkerResult
        Detects hand and extract landmarks and handedness, if a hand is found.
    """
    def __init__(self):
        self.base_options = python.BaseOptions(
            model_asset_path='bundles/hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options,
                                                    num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    
    def detect_landmark_from_file(self, image_path: str) -> HandLandmarkerResult:
        """
        Detects hand and extract landmarks and handedness, if a hand is found.

        Parameters
        ----------
        image_path: str
            relative path to image file.

        Returns
        -------
        HandLandmarkerResult
            Returns an object of type HandLandmarkerResult, where handedness and 21 landmarks are included.
        """
        image = mp.Image.create_from_file(image_path)
        return self.detector.detect(image)
