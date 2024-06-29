import os
import csv
import cv2
from lib.handLandmarker import HandLandmarker
from lib.util import printProgressBar
import mediapipe as mp

class DatasetAnnotator:
    def __init__(self):
        self.handLandmarker = HandLandmarker()

    def annotate(self):
        items = os.listdir('dataset2')
        classes = [item for item in items if os.path.isdir(
            os.path.join('dataset2', item))]
        keypoints_directory_path = os.path.join(os.getcwd() + '/keypoints2')
        os.makedirs(f"{keypoints_directory_path}/", exist_ok=True)
        header = [f"Keypoint_{i}" for i in range(21)]
        for class_label in classes:
            if os.path.isfile(f'{keypoints_directory_path}/{class_label.upper()}.csv'):
                continue
            rows = []
            total_files = len(os.listdir(f'dataset2/{class_label}'))
            progress_prefix = f'Extracting landmarks for class ({class_label.upper()}):'
            for file_index, file_name in enumerate(os.listdir(f'dataset2/{class_label}')):
                if '.DS_Store' in file_name:
                    continue
                img = cv2.imread(f'dataset2/{class_label}/{file_name}')
                height, width = img.shape[:2]
                
                detection_result = self.handLandmarker.detect_landmark_from_file(
                    f'dataset2/{class_label}/{file_name}')
                if (len(detection_result.hand_landmarks) > 0):
                    new_row = []
                    for _, normalizedLandmark in enumerate(detection_result.hand_landmarks[0]):
                        new_row.append(
                            [normalizedLandmark.x, normalizedLandmark.y])
                    rows.append(new_row)
                printProgressBar(file_index + 1, total_files,
                                 prefix=progress_prefix, suffix='Complete', length=50)

            with open(f"{keypoints_directory_path}/{class_label}.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(rows)

    def annotate_new(self):
        mp_hands = mp.solutions.hands
        items = os.listdir('dataset')
        classes = [item for item in items if os.path.isdir(
            os.path.join('dataset', item))]
        keypoints_directory_path = os.path.join(os.getcwd() + '/keypoints')
        os.makedirs(f"{keypoints_directory_path}/", exist_ok=True)
        header = [f"Keypoint_{i}" for i in range(21)]
        for class_label in classes:
            rows = []
            total_files = len(os.listdir(f'dataset/{class_label}'))
            progress_prefix = f'Extracting landmarks for class ({class_label.upper()}):'
            for file_index, file_name in enumerate(os.listdir(f'dataset/{class_label}')):
                if '.DS_Store' in file_name:
                    continue
                with mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.5) as hands:
                    results = hands.process(cv2.imread(f'dataset/{class_label}/{file_name}'))
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.append([lm.x, lm.y])
                        rows.append(landmarks)
                printProgressBar(file_index + 1, total_files,
                                 prefix=progress_prefix, suffix='Complete', length=50)

            with open(f"{keypoints_directory_path}/{class_label}.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(rows)
