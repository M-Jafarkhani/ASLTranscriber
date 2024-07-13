import cv2
import mediapipe as mp
from lib.landmarkDetector import LandmarkDetector


class SignTranscriber:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.detector = LandmarkDetector()
        self.cap = cv2.VideoCapture(0)

    def transcribe(self):
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
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.append([lm.x, lm.y, lm.z])
                        landmarks_list.append(landmarks)
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        result = self.detector.predict_with_classifier(
                            landmarks)
                        #result = f'X = {round(hand_landmarks.landmark[4].x,2)},Y = {round(hand_landmarks.landmark[4].y,2)},Z = {round(hand_landmarks.landmark[4].z,2)}'
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
