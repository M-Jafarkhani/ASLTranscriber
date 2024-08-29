import os
from typing import Any, Optional
from lib.util import *
import pandas as pd


class FeaturesExtractor:
    """
    A Python class for extracting features from annotated data

    ...

    Attributes
    ----------
    keypoints_folder : str
        The folder name which corresponds to extracted keypoints data location.

    features_folder : str
        The folder name which corresponds to extarcted features data location.

    features_file : str
        The file name which corresponds to extarcted features data location.

    df : pd.DataFrame()
        Main dataframe that we store extracted features.

    Methods
    -------
    extract() -> None:
        Starts the feature exraction process.

    get_features(self, handedness: str, landmarks: Optional[int], label_index: Optional[int] = None) -> dict[str, Any | None]:
        Computes features for one image/input.
    """

    def __init__(self) -> None:
        """
        Inits the attributes, including setting keypoints_folder to 'keypoints', features_folder to 'features'
        and features_file to 'data' as default values.

        Parameters
        ----------
        None
        """
        self.keypoints_folder = 'keypoints'
        self.features_folder = 'features'
        self.features_file = 'data'
        self.df = pd.DataFrame()

    def extract(self) -> None:
        """
        In a loop over pre-defined labels, we compute features and save them into one csv file.

        Parameters
        ----------
        None
        """
        for label_index, label in LABELS.items():
            progress_prefix = f'Calculating Features for class ({label.upper()}):'
            file_path = f'{self.keypoints_folder}/{label.lower()}.csv'
            raw_df = pd.read_csv(file_path)
            row_count, _ = raw_df.shape
            for i, row in raw_df.iterrows():
                new_row = self.get_features(
                    row[0], row[1:], label_index)
                self.df = pd.concat(
                    [self.df, pd.DataFrame([new_row])], ignore_index=True)
                printProgressBar(
                    i + 1, row_count, prefix=progress_prefix, suffix='Complete', length=50)
        if not os.path.exists(f'{self.features_folder}'):
            os.makedirs(f'{self.features_folder}')
        self.df.to_csv(
            f'{self.features_folder}/{self.features_file}.csv', index=False)

    def get_features(self, handedness: str, landmarks: Optional[int], label_index: Optional[int] = None) -> dict[str, any]:
        """
        Computes features for one image/input. 

        Parameters
        ----------
        handedness: str
            'R' for right-hand or 'L' for left-hand.

        landmarks: Optional[int]
            List of 21 landmarks, with x, y and z coordinates.

        label_index: Optional[int]
            Index of each label.

        Returns
        -------
        dict[str, any]
            A dataframe row  with computed features.
        """
        new_row = {'label': label_index}
        new_row.update(get_palm_state(handedness, landmarks))
        new_row.update(get_distance(landmarks))
        new_row.update(get_angles(landmarks))
        return new_row
