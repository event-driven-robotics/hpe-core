
import numpy as np
import os

from time import time


# DVS camera
H36M_VIDEO_HEIGHT = 1000
H36M_VIDEO_WIDTH = 1000
H36M_DVS_WIDTH = '346'
H36M_DVS_HEIGHT = '260'
# DHP19_CAM_FRAME_EVENTS_NUM = 7500  # fixed number of events used in DHP19 for generating event frames
subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
all_cameras = {1: '54138969', 2: '55011271', 3: '58860488', 4: '60457274'}

# map from indices to body parts for h36m
H36M_BODY_PARTS_rev = {0: 'PelvisC', 1: 'PelvisR', 2: 'KneeR', 3: 'AnkleR', 4: 'ToeR', 5: 'ToeROther', 6: 'PelvisL',
                       7: 'KneeL', 8: 'AnkleL', 9: 'ToeR', 10: 'ToeROther', 11: 'Spine', 12: 'SpineM', 13: 'Neck',
                       14: 'Head', 15: 'HeadOther', 16: 'NextAgain', 17: 'ShoulderL', 18: 'ElbowL', 19: 'WristL',
                       20: 'WristLAgain', 21: 'ThumbL', 22: 'BOHR', 23: 'BOHRAgain', 24: 'NeckAgainAgain',
                       25: 'ShoulderR', 26: 'ElbowR', 27: 'WristR', 28: 'WristAgain', 29: 'ThumbR', 30: 'BOHL',
                       31: 'BOHLAgain'}

H36M_BODY_PARTS = {v: k for k, v in H36M_BODY_PARTS_rev.items()}

# Following markers coincide with each other
# 0 = 11
# 13 = 16 = 24
# 19 = 20
# 22 = 23
# 26 = 27
# 30 = 31

# {
# 'head': 0,
# 'shoulderR': 1,
# 'shoulderL': 2,
# 'elbowR': 3,
# 'elbowL': 4,
# 'hipL': 5,
# 'hipR': 6,
# 'handR': 7,
# 'handL': 8,
# 'kneeR': 9,
# 'kneeL': 10,
# 'footR': 11,
# 'footL': 12
# }

H36M_TO_DHP19_INDICES = np.array([
    # TODO: compute head
    [14, 0],  # head
    [25, 1],  # shoulderR
    [17, 2],  # shoulderL
    [26, 3],  # elbowR
    [18, 4],  # elbowL
    [6, 5],  # hipL
    [1, 6],  # hipR
    [27, 7],  # handR
    [19, 8],  # handL
    [2, 9],  # kneeR
    [7, 10],  # kneeL
    [3, 11],  # footR
    [8, 12]  # footL
])

DHP19_TO_MOVENET_INDICES = np.array([
    # TODO: fix to be similar to the previous one.
    [0, 0],  # head
    [1, 2],  # lshoulder
    [2, 1],  # rshoulder
    [3, 4],  # lelbow
    [4, 3],  # relbow
    [5, 8],  # lwrist
    [6, 7],  # rwrist
    [7, 5],  # lhip
    [8, 6],  # rhip
    [9, 10],  # lknee
    [10, 9],  # rknee
    [11, 12],  # lankle
    [12, 11]  # rankle
])

MOVENET_TO_DHP19_INDICES = DHP19_TO_MOVENET_INDICES[np.argsort(DHP19_TO_MOVENET_INDICES[:,1]),:]

# H36M_TO_HPECORE_SKELETON_MAP = OrderedDict()
# H36M_TO_HPECORE_SKELETON_MAP['head'] = 14
# H36M_TO_HPECORE_SKELETON_MAP['shoulderL'] = 17
# H36M_TO_HPECORE_SKELETON_MAP['shoulderR'] = 25
# H36M_TO_HPECORE_SKELETON_MAP['elbowL'] = 18
# H36M_TO_HPECORE_SKELETON_MAP['elbowR'] = 26
# H36M_TO_HPECORE_SKELETON_MAP['handL'] = 19
# H36M_TO_HPECORE_SKELETON_MAP['handR'] = 27
# H36M_TO_HPECORE_SKELETON_MAP['hipL'] = 6
# H36M_TO_HPECORE_SKELETON_MAP['hipR'] = 1
# H36M_TO_HPECORE_SKELETON_MAP['kneeL'] = 7
# H36M_TO_HPECORE_SKELETON_MAP['kneeR'] = 2
# H36M_TO_HPECORE_SKELETON_MAP['footL'] = 8
# H36M_TO_HPECORE_SKELETON_MAP['footR'] = 3

H36M_TO_HPECORE_SKELETON_MAP = [
    14,  # head
    25,  # shoulderR
    17,  # shoulderL
    26,  # elbowR
    18,  # elbowL
    6,  # hipL
    1,  # hipR
    27,  # handR
    19,  # handL
    2,  # kneeR
    7,  # kneeL
    3,  # footR
    8  # footL
]


def h36m_to_hpecore_skeleton(pose):
    return pose[H36M_TO_HPECORE_SKELETON_MAP, :]


def h36m_to_dhp19(pose):
    return pose[H36M_TO_DHP19_INDICES[:, 0], :]


def dhp19_to_movenet(pose):
    return pose[DHP19_TO_MOVENET_INDICES[:, 1], :]

def hpecore_to_movenet(pose):
    return dhp19_to_movenet(pose)

def movenet_to_dhp19(pose):
    return pose[MOVENET_TO_DHP19_INDICES[:, 0],:]

def movenet_to_hpecore(pose):
    return movenet_to_dhp19(pose)

def dhp19_to_h36m(pose):
    # TODO
    pass

def get_h36m_body_parts(pose):
    inv_map = {v: k for k, v in pose.items()}
    return inv_map


# def openpose_to_dhp19(pose_op):
#     # TODO: compute dhp19's head joints from openpose
#     return pose_op[OPENPOSE_TO_DHP19_INDICES[:, 0], :]

def writer(directory, datalines, infolines):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # datafile.log
    dataFile = directory + '/data.log'
    with open(dataFile, 'w') as f:
        for line in datalines:
            f.write("%s\n" % line)
    # info.log

    infoFile = directory + '/info.log'
    with open(infoFile, 'w') as f:
        for line in infolines:
            f.write("%s\n" % line)
