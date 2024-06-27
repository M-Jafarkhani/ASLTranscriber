import cv2
import mediapipe as mp
from lib.handLandmarker import HandLandmarker
from lib.landmarkClassifier import LandmarkClassifier


class SignTranscriber:
    def __init__(self):
        self.handLandmarker = HandLandmarker()

    def transcribe(self):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        classifier = LandmarkClassifier()
        cap = cv2.VideoCapture(0)

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
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
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        result = classifier.new_predict(landmarks)
                        wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
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

    def transcribe_2(self):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        classifier = LandmarkClassifier()
        cap = cv2.VideoCapture(0)

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                detection_result = self.handLandmarker.detect_landmark_from_data(
                    image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if (len(detection_result.hand_landmarks) > 0):
                    data = []
                    for _, normalizedLandmark in enumerate(detection_result.hand_landmarks[0]):
                        data.append(
                            [normalizedLandmark.x, normalizedLandmark.y])
                    mp_drawing.draw_landmarks(
                        image, results.multi_hand_landmarks.landmarks[0], mp_hands.HAND_CONNECTIONS)
                    result = classifier.predict(data)
                    wrist_landmark = detection_result.hand_landmarks[0][mp_hands.HandLandmark.WRIST]
                    h, w, c = image.shape
                    wrist_coords = (int(wrist_landmark.x * w),
                                    int(wrist_landmark.y * h))
                    cv2.putText(image, result, wrist_coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('ASL Transcriber', image)
                else:
                    cv2.imshow('ASL Transcriber', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
