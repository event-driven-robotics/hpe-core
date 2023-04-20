import cv2
import numpy as np

from .parsing import JOINT_NOT_PRESENT

ALGO_COLORS = {
    # 'openpose_rgb': 'blue',
    # 'openpose_pim': 'green',
    # 'movenet_cam2': 'orange',
    # 'movenet_cam-24': 'red',
    'movenet_eF': (122.0/255,1.0/255,119.0/255),
    'movenet_wo_finetune': (251.0/255,180.0/255,185.0/255),
    'gl-hpe':(37.0/255,52.0/255,148.0/255),
    'movenet': (197.0/255, 27.0/255, 138.0/255),
    'movenet_wo_pretrain': (247.0/255, 104.0/255, 161.0/255),
    'openpose_rgb': (44.0/255,127.0/255,184.0/255),
    'openpose_pim': (194.0/255,230.0/255,153.0/255),
    'movenet_cam2': (255.0/255, 255.0/255, 255.0/255),
    'movenet_cam-24': (197.0/255, 27.0/255, 138.0/255),
    'movenet_rgb':(37.0/255,52.0/255,148.0/255)
}

OTHER_COLORS = ['black', 'violet','grey']

RENAMING = {'movenet_cam-24': 'moveEnet',
            'movenet_eF': 'movenet_fixedCount',
            'movenet': 'moveEnet',
            'gl-hpe':'liftmono-hpe'
            }


def plot_poses(img, pose_gt, pose_pred):
    for joint_gt, joint_pred in zip(pose_gt, pose_pred):
        cv2.circle(img, (joint_gt[1], joint_gt[0]), 5, (0, 255, 0), cv2.FILLED)
        if _is_no_joint(joint_pred):
            continue
        cv2.circle(img, (joint_pred[0], joint_pred[1]), 5, (0, 0, 255), cv2.FILLED)
        cv2.imshow('',img)
        cv2.waitKey(1)


###########
# PRIVATE #
###########

def _is_no_joint(joint):
    return np.sum(joint == JOINT_NOT_PRESENT) == 2
