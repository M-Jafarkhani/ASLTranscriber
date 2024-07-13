from scipy.spatial import distance
import ast
import math

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
    25: 'Z',
    26: '1',
    27: '2',
    28: '3',
    29: '4',
    30: '5',
    31: '6',
    32: '7',
    33: '8',
    34: '9',
    35: '10'
}

def calculate_angle(P1, P2, P3):
    if type(P1) == str:
        P1 = ast.literal_eval(P1)
        P2 = ast.literal_eval(P2)
        P3 = ast.literal_eval(P3)

    x1 = P1[0]
    y1 = P1[1]

    x2 = P2[0]
    y2 = P2[1]

    x3 = P3[0]
    y3 = P3[1]

    v1 = (x2 - x1, y2 - y1)
    v2 = (x3 - x2, y3 - y2)

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    cos_theta = max(-1, min(1, cos_theta))
    angle_radians = math.acos(cos_theta)
    

    return math.degrees(angle_radians)


def euclidean_distance(landmark_1, landmark_2):
    if type(landmark_1) == str:
        landmark_1 = ast.literal_eval(landmark_1)
        landmark_2 = ast.literal_eval(landmark_2)

    return distance.euclidean([landmark_1[0], landmark_1[1]], [landmark_2[0], landmark_2[1]])

def get_x(landmark):
    if type(landmark) == str:
        landmark = ast.literal_eval(landmark)

    return landmark[0]

def get_y(landmark):
    if type(landmark) == str:
        landmark = ast.literal_eval(landmark)

    return landmark[1]

def get_distance(landmark):
    return {'d_4_0': euclidean_distance(landmark[4], landmark[0]),
            'd_8_0': euclidean_distance(landmark[8], landmark[0]),
            'd_12_0': euclidean_distance(landmark[12], landmark[0]),
            'd_16_0': euclidean_distance(landmark[16], landmark[0]),
            'd_20_0': euclidean_distance(landmark[20], landmark[0]),
            'd_4_8': euclidean_distance(landmark[4], landmark[8]),
            'd_8_12': euclidean_distance(landmark[8], landmark[12]),
            'd_12_16': euclidean_distance(landmark[12], landmark[16]),
            'd_16_20': euclidean_distance(landmark[16], landmark[20]),
            
            'd_3_7': euclidean_distance(landmark[3], landmark[7]),
            'd_6_2': euclidean_distance(landmark[6], landmark[2]),
            'd_5_1': euclidean_distance(landmark[5], landmark[1]),

            'd_7_11': euclidean_distance(landmark[7], landmark[11]),
            'd_6_10': euclidean_distance(landmark[6], landmark[10]),
            'd_5_9': euclidean_distance(landmark[5], landmark[9]),

            'd_11_15': euclidean_distance(landmark[11], landmark[15]),
            'd_10_14': euclidean_distance(landmark[10], landmark[14]),
            'd_9_13': euclidean_distance(landmark[9], landmark[13]),

            'd_15_19': euclidean_distance(landmark[15], landmark[19]),
            'd_14_18': euclidean_distance(landmark[14], landmark[18]),
            'd_13_17': euclidean_distance(landmark[13], landmark[17])
            }


def get_angles(landmark):
    return {'a_0_1_2': calculate_angle(landmark[0], landmark[1], landmark[2]),
            'a_1_2_3': calculate_angle(landmark[1], landmark[2], landmark[3]),
            'a_2_3_4': calculate_angle(landmark[2], landmark[3], landmark[4]),
            'a_0_5_6': calculate_angle(landmark[0], landmark[5], landmark[6]),
            'a_5_6_7': calculate_angle(landmark[5], landmark[6], landmark[7]),
            'a_6_7_8': calculate_angle(landmark[6], landmark[7], landmark[8]),
            'a_6_5_9': calculate_angle(landmark[6], landmark[5], landmark[9]),
            'a_5_9_10': calculate_angle(landmark[5], landmark[9], landmark[10]),
            'a_9_10_11': calculate_angle(landmark[9], landmark[10], landmark[11]),
            'a_10_11_12': calculate_angle(landmark[10], landmark[11], landmark[12]),
            'a_9_13_14': calculate_angle(landmark[9], landmark[13], landmark[14]),
            'a_13_14_15': calculate_angle(landmark[13], landmark[14], landmark[15]),
            'a_14_15_16': calculate_angle(landmark[14], landmark[15], landmark[16]),
            'a_14_13_17': calculate_angle(landmark[14], landmark[13], landmark[17]),
            'a_13_17_18': calculate_angle(landmark[13], landmark[17], landmark[18]),
            'a_17_18_19': calculate_angle(landmark[17], landmark[18], landmark[19]),
            'a_18_19_20': calculate_angle(landmark[18], landmark[19], landmark[20]),
            'a_0_17_18': calculate_angle(landmark[0], landmark[17], landmark[18])}

def get_points(landmark):
    return {'x_0': get_x(landmark[0]),
            'y_0': get_y(landmark[0]),

            'x_1': get_x(landmark[1]),
            'y_1': get_y(landmark[1]),
            
            'x_2': get_x(landmark[2]),
            'y_2': get_y(landmark[2]),
            
            'x_3': get_x(landmark[3]),
            'y_3': get_y(landmark[3]),
            
            'x_4': get_x(landmark[4]),
            'y_4': get_y(landmark[4]),
            
            'x_5': get_x(landmark[5]),
            'y_5': get_y(landmark[5]),
            
            'x_6': get_x(landmark[6]),
            'y_6': get_y(landmark[6]),
            
            'x_7': get_x(landmark[7]),
            'y_7': get_y(landmark[7]),
            
            'x_8': get_x(landmark[8]),
            'y_8': get_y(landmark[8]),
            
            'x_9': get_x(landmark[9]),
            'y_9': get_y(landmark[9]),
            
            'x_10': get_x(landmark[10]),
            'y_10': get_y(landmark[10]),
            
            'x_11': get_x(landmark[11]),
            'y_11': get_y(landmark[11]),
            
            'x_12': get_x(landmark[12]),
            'y_12': get_y(landmark[12]),

            'x_13': get_x(landmark[13]),
            'y_13': get_y(landmark[13]),

            'x_14': get_x(landmark[14]),
            'y_14': get_y(landmark[14]),
            
            'x_15': get_x(landmark[15]),
            'y_15': get_y(landmark[15]),
            
            'x_16': get_x(landmark[16]),
            'y_16': get_y(landmark[16]),

            'x_17': get_x(landmark[17]),
            'y_17': get_y(landmark[17]),

            'x_18': get_x(landmark[18]),
            'y_18': get_y(landmark[18]),

            'x_19': get_x(landmark[19]),
            'y_19': get_y(landmark[19]),

            'x_20': get_x(landmark[20]),
            'y_20': get_y(landmark[20])

            }

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
