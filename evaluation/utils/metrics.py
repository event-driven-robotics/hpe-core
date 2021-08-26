
import numpy as np


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


def print_mpjpe(mpjpe, keypoints_str):
    for ei, error in enumerate(mpjpe):
        print(f'{keypoints_str[ei]}: {error}')
