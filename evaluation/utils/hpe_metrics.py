import json
import numpy as np

# map from body parts to indices for dhp19
DHP19_BODY_PARTS = {
    'head': 0,
    'shoulderR': 1,
    'shoulderL': 2,
    'elbowR': 3,
    'elbowL': 4,
    'hipL': 5,
    'hipR': 6,
    'handR': 7,
    'handL': 8,
    'kneeR': 9,
    'kneeL': 10,
    'footR': 11,
    'footL': 12
}

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

OPENPOSE_TO_DHP19_INDICES = np.array([
    # TODO: compute head
    [0, 0],  # head
    [2, 1],  # shoulderR
    [5, 2],  # shoulderL
    [3, 3],  # elbowR
    [6, 4],  # elbowL
    [12, 5],  # hipL
    [9, 6],  # hipR
    [4, 7],  # handR
    [7, 8],  # handL
    [10, 9],  # kneeR
    [13, 10],  # kneeL
    [11, 11],  # footR
    [14, 12]  # footL
])


# TODO: what if a gt joint is not present in a frame?
def compute_pck(predicted_joints, gt_joints, threshold, head_size=None):
    """
    Computes PCK (Percentage of Correct Keypoints)
    """
    # compute euclidean distances between joints
    distances = np.linalg.norm(predicted_joints - gt_joints, axis=2)

    # normalize distances according to head size
    if head_size:
        distances = distances / head_size

    # compute correct keypoints
    correct_keypoints = (distances <= threshold).astype(int)

    # compute pck
    pck = np.sum(correct_keypoints, axis=0) / correct_keypoints.shape[0]

    return pck


def compute_mpjpe(predicted_joints, gt_joints):
    """
    Computes MPJPE (Mean Per Joint Position Error)
    """
    # compute euclidean distances between joints
    distances = np.linalg.norm(predicted_joints - gt_joints, axis=2)

    # TODO: what if a gt joint is not present in a frame? get the number of gt joints by
    # gt_num = np.sum(gt_joints == 0)
    mpje = np.sum(distances, axis=0) / len(distances)

    return mpje


def openpose_to_dhp19(pose_op):
    # TODO: compute dhp19's head joints from openpose
    return (pose_op[OPENPOSE_TO_DHP19_INDICES[:, 0], :]).astype(int)


def parse_openpose_keypoints_json(json_path):
    with open(json_path, 'r') as fp:
        json_dict = json.load(fp)

    # if there are no detections, return zero metrix
    if len(json_dict['people']) == 0:
        return np.zeros((len(OPENPOSE_BODY_PARTS_25), 2))

    # TODO: return all poses from openpose?
    # for now assume there's at most one predicted pose
    keypoints = np.array(json_dict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))

    # return 2d coordinates only, remove prediction confidence
    return keypoints[:, :-1]
