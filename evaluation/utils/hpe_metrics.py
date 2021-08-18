
import numpy as np

# list indices of openpose joints
OPENPOSE_BODY_PARTS_25 = {
    0: "Nose",
    1:  "Neck",
    2:  "RShoulder",
    3:  "RElbow",
    4:  "RWrist",
    5:  "LShoulder",
    6:  "LElbow",
    7:  "LWrist",
    8:  "MidHip",
    9:  "RHip",
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


# pck
def compute_pck(predicted_joints, gt_joints, threshold, head_size=None):
    """
    Computes PCK (Percentage of Correct Keypoints)
    """
    # compute ditances between joints
    distances = np.linalg.norm(predicted_joints - gt_joints, axis=2)

    # normalize distances according to head size
    if head_size:
        distances = distances / head_size

    # compute correct keypoints
    correct_keypoints = (distances <= threshold).astype(int)

    # compute pck
    pck = np.sum(correct_keypoints, axis=0) / correct_keypoints.shape[0]

    return pck
