
import cv2
import numpy as np

from .parsing import JOINT_NOT_PRESENT


def plot_poses(img, pose_gt, pose_pred):

    for joint_gt, joint_pred in zip(pose_gt, pose_pred):
        cv2.circle(img, (joint_gt[1], joint_gt[0]), 5, (0, 255, 0), cv2.FILLED)
        if _is_no_joint(joint_pred):
            continue
        cv2.circle(img, (joint_pred[0], joint_pred[1]), 5, (0, 0, 255), cv2.FILLED)


###########
# PRIVATE #
###########

def _is_no_joint(joint):
    return np.sum(joint == JOINT_NOT_PRESENT) == 2
