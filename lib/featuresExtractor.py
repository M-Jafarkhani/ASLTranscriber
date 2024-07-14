import os
from lib.util import *
import pandas as pd


class FeaturesExtractor:
    def __init__(self):
        self.keypoints_folder = 'keypoints'
        self.features_folder = 'features'
        self.features_file = 'data'
        self.df = pd.DataFrame()

    def extract(self):
        for label_index, label in LABELS.items():
            progress_prefix = f'Calculating Features for class ({label.upper()}):'
            file_path = f'{self.keypoints_folder}/{label.lower()}.csv'
            raw_df = pd.read_csv(file_path)
            row_count, _ = raw_df.shape
            for i, row in raw_df.iterrows():
                new_row = self.get_features(row, label_index)
                self.df = pd.concat(
                    [self.df, pd.DataFrame([new_row])], ignore_index=True)
                printProgressBar(
                    i + 1, row_count, prefix=progress_prefix, suffix='Complete', length=50)
        if not os.path.exists(f'{self.features_folder}'):
            os.makedirs(f'{self.features_folder}')
        self.df.to_csv(
            f'{self.features_folder}/{self.features_file}.csv', index=False)

    def get_features(self, landmarks, label_index=None):
        new_row = {'label': label_index}
        new_row.update(get_distance(landmarks))
        new_row.update(get_angles(landmarks))
        return new_row
