import numpy as np
import os

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
                       20: 'WristLAgain', 21: 'ThumbL', 22: 'BOHR', 23: 'BOHRAgain',  24: 'NeckAgainAgain',
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


def h36m_to_dhp19(pose):
    return pose[H36M_TO_DHP19_INDICES[:, 0], :]


# TODO

def get_h36m_body_parts(pose):
    inv_map = {v: k for k, v in pose.items()}
    return inv_map
# def openpose_to_dhp19(pose_op):
#     # TODO: compute dhp19's head joints from openpose
#     return pose_op[OPENPOSE_TO_DHP19_INDICES[:, 0], :]

def writer(directory,datalines,infolines):

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

#
# class Dhp19EventsIterator:
#
#     # TODO: add param for overlapping?
#     # def __init__(self, data, cam_id, window_size=DHP19_FRAME_EVENTS_NUM, stride=None):
#     def __init__(self, data, cam_id, window_size=DHP19_CAM_FRAME_EVENTS_NUM):
#
#         self.timestamps = data['out']['extra']['ts']  # array containing timestamps of events from all cameras
#
#         self.events = data['out']['data'][f'cam{cam_id}']['dvs']  # events specific to selected camera
#
#         # events location indices follow matlab indexing convention, i.e. they start from 1 instead of 0
#         self.events['x'] = self.events['x'] - 1
#         self.events['y'] = self.events['y'] - 1
#
#         # events x indices are shifted by sensor_width * camera id
#         self.events['x'] = self.events['x'] - DHP19_SENSOR_WIDTH * cam_id
#
#         # events are sampled from all cameras, thus the actual window size is the desired input one (representing the
#         # desired number of frames from a single camera) multiplied by the number of cameras (for an explanation, see
#         # Section 4.1 of paper "DHP19: Dynamic Vision Sensor 3D Human Pose Dataset")
#         self.window_size = window_size * DHP19_CAM_NUM
#
#         self.curr_ind = 0
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         return int(np.ceil(len(self.timestamps) / self.window_size))
#
#     def __next__(self):
#         if self.curr_ind == -1:
#             raise StopIteration
#
#         end_ind = self.curr_ind + self.window_size
#
#         # select events from the specified camera with timestamps within the current window
#         window_timestamps = self.timestamps[self.curr_ind:end_ind]
#         event_indices = np.isin(self.events['ts'], window_timestamps)
#         data = np.concatenate((np.reshape(self.events['ts'][event_indices], (-1, 1)),
#                                np.reshape(self.events['x'][event_indices], (-1, 1)),
#                                np.reshape(self.events['y'][event_indices], (-1, 1)),
#                                np.reshape(self.events['pol'][event_indices], (-1, 1))), axis=1, dtype=np.float64)
#
#         if end_ind >= self.timestamps.shape[0]:
#             self.curr_ind = -1
#         else:
#             self.curr_ind = end_ind
#
#         return data
