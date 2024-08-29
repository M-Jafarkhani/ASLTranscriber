import os
import csv
from lib.handLandmarker import HandLandmarker
from lib.util import printProgressBar


class DatasetAnnotator:
    """
    A Python class for annotating train data

    ...

    Attributes
    ----------
    handLandmarker : HandLandmarker
        Main class imported from mediapipe, which is responsible for hand-detection and landmark extraction
    
    dataset_folder : str
        The folder name which corresponds to train data location
    
    keypoints_folder : str
        The folder name which corresponds to extracted keypoints data location

    Methods
    -------
    annotate() -> None:
        Starts the data annotation process.
    """
        
    def __init__(self) -> None:
        """
        Inits the attributes, including setting dataset_folder to 'dataset' and keypoints_folder to 'keypoints' 

        Parameters
        ----------
        None
        """
        self.handLandmarker = HandLandmarker()
        self.dataset_folder = 'dataset'
        self.keypoints_folder = 'keypoints'

    def annotate(self) -> None:
        """
        The main annotation process. The train data are expected to be separated into their corresponding folders,
        where each folder name corresponds to their label.

        The handedness and 21 landmarks for each class are extratced and saved into one csv file.

        Parameters
        ----------
        None
        """
        items = os.listdir(f'{self.dataset_folder}')
        classes = [item for item in items if os.path.isdir(
            os.path.join(f'{self.dataset_folder}', item))]
        keypoints_directory_path = os.path.join(
            os.getcwd() + f'/{self.keypoints_folder}')
        os.makedirs(f"{keypoints_directory_path}/", exist_ok=True)
        header = [f"Keypoint_{i}" for i in range(21)]
        header.insert(0, "Handedness")
        for class_label in classes:
            if os.path.isfile(f'{keypoints_directory_path}/{class_label.upper()}.csv'):
                continue
            rows = []
            total_files = len(os.listdir(
                f'{self.dataset_folder}/{class_label}'))
            progress_prefix = f'Extracting landmarks for class ({class_label.upper()}):'
            for file_index, file_name in enumerate(os.listdir(f'{self.dataset_folder}/{class_label}')):
                if '.DS_Store' in file_name:
                    continue
                detection_result = self.handLandmarker.detect_landmark_from_file(
                    f'{self.dataset_folder}/{class_label}/{file_name}')
                if (len(detection_result.hand_landmarks) > 0):
                    new_row = [detection_result.handedness[0]
                               [0].display_name[0]]
                    for _, normalizedLandmark in enumerate(detection_result.hand_landmarks[0]):
                        new_row.append(
                            [normalizedLandmark.x, normalizedLandmark.y, normalizedLandmark.z])
                    rows.append(new_row)
                printProgressBar(file_index + 1, total_files,
                                 prefix=progress_prefix, suffix='Complete', length=50)

            with open(f"{keypoints_directory_path}/{class_label}.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(rows)
