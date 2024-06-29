import os
from lib.util import calculate_angle, calculate_distance, printProgressBar
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
            progress_prefix = f'Calculating Distance for class ({label.upper()}):'
            file_path = f'keypoints/{label.lower()}.csv'
            raw_df = pd.read_csv(file_path)
            row_count, _ = raw_df.shape
            for i, row in raw_df.iterrows():
                distances = calculate_distance(row)
                new_row = {'label': label_index,
                        #    'distance_4_0': distances[0],
                        #    'distance_8_0': distances[1],
                        #    'distance_12_0': distances[2],
                        #    'distance_16_0': distances[3],
                        #    'distance_20_0': distances[4],
                        #    'distance_4_8': distances[5],
                        #    'distance_8_12': distances[6],
                        #    'distance_12_16': distances[7],
                        #    'distance_16_20': distances[8],
                           'angle_0_1_2': calculate_angle(row[0],row[1],row[2]),
                           'angle_1_2_3': calculate_angle(row[1],row[2],row[3]),
                           'angle_2_3_4': calculate_angle(row[2],row[3],row[4]),
                           'angle_0_5_6': calculate_angle(row[0],row[5],row[6]),
                           'angle_5_6_7': calculate_angle(row[5],row[6],row[7]),
                           'angle_6_7_8': calculate_angle(row[6],row[7],row[8]),
                           'angle_6_5_9': calculate_angle(row[6],row[5],row[9]),
                           'angle_5_9_10': calculate_angle(row[5],row[9],row[10]),
                           'angle_9_10_11': calculate_angle(row[9],row[10],row[11]),
                           'angle_10_11_12': calculate_angle(row[10],row[11],row[12]),
                           'angle_9_13_14': calculate_angle(row[9],row[13],row[14]),
                           'angle_13_14_15': calculate_angle(row[13],row[14],row[15]),
                           'angle_14_15_16': calculate_angle(row[14],row[15],row[16]),
                           'angle_14_13_17': calculate_angle(row[14],row[13],row[17]),
                           'angle_13_17_18': calculate_angle(row[13],row[17],row[18]),
                           'angle_17_18_19': calculate_angle(row[17],row[18],row[19]),
                           'angle_18_19_20': calculate_angle(row[18],row[19],row[20]),
                           'angle_0_17_18': calculate_angle(row[0],row[17],row[18])
                           }
                self.df = pd.concat(
                    [self.df, pd.DataFrame([new_row])], ignore_index=True)
                printProgressBar(i + 1, row_count,prefix=progress_prefix, suffix='Complete', length=50)        
        if not os.path.exists('distance'):
            os.makedirs('distance')
        self.df.to_csv(r'distance/data.csv', index=False)
