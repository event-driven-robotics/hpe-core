
import json
import numpy as np


# map from indices to body parts for openpose
OPENPOSE_BODY_PARTS_25 = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "MidHip",
    9: "RHip",
    10: "RKnee",
    11: "RAnkle",
    12: "LHip",
    13: "LKnee",
    14: "LAnkle",
    15: "REye",
    16: "LEye",
    17: "REar",
    18: "LEar",
    19: "LBigToe",
    20: "LSmallToe",
    21: "LHeel",
    22: "RBigToe",
    23: "RSmallToe",
    24: "RHeel",
    25: "Background"
}

JOINT_NOT_PRESENT = np.zeros(2)


def parse_openpose_keypoints_json(json_path):
    """

    """
    with open(json_path, 'r') as fp:
        json_dict = json.load(fp)

    # if there are no detections, return zero metrix
    if len(json_dict['people']) == 0:
        return np.zeros((len(OPENPOSE_BODY_PARTS_25), 2), dtype=int)

    # TODO: return all poses from openpose?
    # for now assume there's at most one predicted pose
    keypoints = np.array(json_dict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))

    # return 2d coordinates only, remove prediction confidence
    return keypoints[:, :-1]
