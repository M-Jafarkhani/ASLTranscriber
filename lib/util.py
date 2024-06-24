from scipy.spatial import distance
import ast

LABELS = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}


LABELS_BACKUP = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
    26: '0',
    27: '1',
    28: '2',
    29: '3',
    30: '4',
    31: '5',
    32: '6',
    33: '7',
    34: '8',
    35: '9'
}


def euclidean_distance(landmark_1, landmark_2):
    if type(landmark_1) == str:
        landmark_1 = ast.literal_eval(landmark_1)
        landmark_2 = ast.literal_eval(landmark_2)

    return distance.euclidean([landmark_1[0], landmark_1[1]], [landmark_2[0], landmark_2[1]])


def calculate_distance(landmark):
    distance_4_0 = euclidean_distance(landmark[4], landmark[0])
    distance_8_0 = euclidean_distance(landmark[8], landmark[0])
    distance_12_0 = euclidean_distance(landmark[12], landmark[0])
    distance_16_0 = euclidean_distance(landmark[16], landmark[0])
    distance_20_0 = euclidean_distance(landmark[20], landmark[0])

    distance_4_8 = euclidean_distance(landmark[4], landmark[8])
    distance_8_12 = euclidean_distance(landmark[8], landmark[12])
    distance_12_16 = euclidean_distance(landmark[12], landmark[16])
    distance_16_20 = euclidean_distance(landmark[16], landmark[20])

    return [distance_4_0,
            distance_8_0,
            distance_12_0,
            distance_16_0,
            distance_20_0,
            distance_4_8,
            distance_8_12,
            distance_12_16,
            distance_16_20]


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

    if iteration == total:
        print()
