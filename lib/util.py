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


def calculate_angle(P1: any, P2: any, P3: any) -> float:
    """
    Calculates the angle between 3 given landmarks 

    Parameters
    ----------
    P1: any
        First landmark

    P2: any
        Second landmark

    P3: any
        Third landmark        

    Returns
    -------
    float
        The minimum angle between 3 given landmarks.
    """
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
    degree = math.degrees(angle_radians)

    return min(degree, 180-degree)


def euclidean_distance(landmark_1: any, landmark_2: any) -> float:
    """
    Calculates the euclidean distance between 2 given landmarks 

    Parameters
    ----------
    landmark_1: any
        First landmark

    landmark_2: any
        Second landmark       

    Returns
    -------
    float
        The euclidean distance between 2 given landmarks
    """
    if type(landmark_1) == str:
        landmark_1 = ast.literal_eval(landmark_1)
        landmark_2 = ast.literal_eval(landmark_2)

    return distance.euclidean([landmark_1[0], landmark_1[1]], [landmark_2[0], landmark_2[1]])


def get_palm_state(handedness: str, landmarks: any) -> dict[str, int]:
    """
    Calculates the palm state, which indicates whether the palm of the hand is facing the camera or not

    Parameters
    ----------
    handedness: str
        'R' for right hand, 'L' for left hand

    landmarks: any
        Landmarks of the hand       

    Returns
    -------
    dict[str, int]
        A dictionary of the form {'p': x} where x is either 1 or -1. (1 for facing the camera).
    """
    landmark_0 = landmarks[0]
    landmark_5 = landmarks[5]
    landmark_17 = landmarks[17]

    if type(landmark_0) == str:
        landmark_0 = ast.literal_eval(landmark_0)
        landmark_5 = ast.literal_eval(landmark_5)
        landmark_17 = ast.literal_eval(landmark_17)

    vector_1 = (landmark_5[0] - landmark_0[0], landmark_5[1] - landmark_0[1])
    vector_2 = (landmark_17[0] - landmark_0[0], landmark_17[1] - landmark_0[1])

    cross_product = vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]

    if handedness == "R":
        return {'p': -1} if cross_product > 0 else {'p': 1}
    else:
        return {'p': -1} if cross_product < 0 else {'p': 1}


def get_distance(landmark: any) -> dict[str, float]:
    """
    Calculates the euclidean distance groups for one hand

    Parameters
    ----------
    landmarks: any
        Landmarks of the hand       

    Returns
    -------
    dict[str, float]
        A dictionary of the form {'d_P1_P2': D} where P1 and P2 are two landmarks, and D is the distance of P1 and P2.
    """
    return {'d_4_0': euclidean_distance(landmark[4], landmark[0]),
            'd_8_0': euclidean_distance(landmark[8], landmark[0]),
            'd_12_0': euclidean_distance(landmark[12], landmark[0]),
            'd_16_0': euclidean_distance(landmark[16], landmark[0]),
            'd_20_0': euclidean_distance(landmark[20], landmark[0]),
            'd_4_8': euclidean_distance(landmark[4], landmark[8]),
            'd_8_12': euclidean_distance(landmark[8], landmark[12]),
            'd_12_16': euclidean_distance(landmark[12], landmark[16]),
            'd_16_20': euclidean_distance(landmark[16], landmark[20]),
            'd_4_12': euclidean_distance(landmark[4], landmark[12]),
            'd_4_16': euclidean_distance(landmark[4], landmark[16]),
            'd_4_20': euclidean_distance(landmark[4], landmark[20])
            }


def get_angles(landmark: any) -> dict[str, float]:
    """
    Calculates the angle groups for one hand

    Parameters
    ----------
    landmarks: any
        Landmarks of the hand       

    Returns
    -------
    dict[str, float]
        A dictionary of the form {'A_P1_P2_P3': A} where P1, P2 and P3 are 3 landmarks, 
        and A is the angle between them, such that P2 is the center.
    """
    return {'a_4_0_8': calculate_angle(landmark[4], landmark[0], landmark[8]),
            'a_8_0_12': calculate_angle(landmark[8], landmark[0], landmark[12]),
            'a_12_0_16': calculate_angle(landmark[12], landmark[0], landmark[16]),
            'a_16_0_20': calculate_angle(landmark[16], landmark[0], landmark[20])}


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r") -> None:
    """
    This code is copied from Stackoverflow, which helps to print progress bar in the terminal. Available at:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters

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
