import os
from lib.util import calculate_distance
from lib.util import LABELS
import pandas as pd


class DistanceExtractor:
    columns = []
    df = pd.DataFrame()

    def __init__(self):
        self.columns = ['label',
                        'distance_4_0',
                        'distance_8_0',
                        'distance_12_0',
                        'distance_16_0',
                        'distance_20_0',
                        'distance_4_8',
                        'distance_8_12',
                        'distance_12_16',
                        'distance_16_20']
        self.df = pd.DataFrame(columns=self.columns)

    def extract(self):
        for label_index, label in LABELS.items():
            file_path = f'keypoints/{label.lower()}.csv'
            raw_df = pd.read_csv(file_path)
            for _, row in raw_df.iterrows():
                distances = calculate_distance(row)
                new_row = {'label': label_index,
                           'distance_4_0': distances[0],
                           'distance_8_0': distances[1],
                           'distance_12_0': distances[2],
                           'distance_16_0': distances[3],
                           'distance_20_0': distances[4],
                           'distance_4_8': distances[5],
                           'distance_8_12': distances[6],
                           'distance_12_16': distances[7],
                           'distance_16_20': distances[8]}
                self.df = pd.concat(
                    [self.df, pd.DataFrame([new_row])], ignore_index=True)
        if not os.path.exists('distance'):
            os.makedirs('distance')
        self.df.to_csv(r'distance/data.csv', index=False)
