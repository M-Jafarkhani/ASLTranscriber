import cv2
import mediapipe as mp
from lib.landmarkDetector import LandmarkDetector


class SignTranscriber:
    """
    A Python class for transcribing the ASL alphabet or digit

    ...

    Attributes
    ----------
    mp_hands : any
        Function from MediaPipe which detects hands in an image

    mp_drawing : any
        Function to draw Landmarks on an image

    detector : LandmarkDetector
        An instance of class LandmarkDetector which predicts which alphabet or digit from ASL are in the image

    cap : VideoCapture
        Object for opening the camera and capture image

    Methods
    -------
    transcribe() -> None:
        Transcribe the ASL hand gesture from the video input 
    """
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.detector = LandmarkDetector()
        self.cap = cv2.VideoCapture(0)

    def transcribe(self) -> None:
        """
        The main transcribtion process which opens the camera, detects the hand and landmarks, and 
        predicts the alphabet or digit from ASL. The default approach is RFD. If you want to switch 
        to DNN apprach, change the 'predict_with_rf' to 'predict_with_dnn' on line 67.

        Parameters
        ----------
        None
        """
        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                landmarks_list = []
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.append([lm.x, lm.y, lm.z])
                        landmarks_list.append(landmarks)
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        result = self.detector.predict_with_dnn(
                            handedness.classification[0].label[0], landmarks)
                        wrist_landmark = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        h, w, _ = image.shape
                        wrist_coords = (int(wrist_landmark.x * w),
                                        int(wrist_landmark.y * h))
                        cv2.putText(image, result, wrist_coords,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow('ASL Transcriber', image)
                else:
                    cv2.imshow('ASL Transcriber', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
