
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

def h36m_to_movenet(pose):
    return dhp19_to_movenet(pose)

def movenet_to_dhp19(pose):
    pass


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


class H36mIterator:
    def __init__(self, data, data_skl):
        # TODO: add return of skeleton

        self.events_ts = data['ts']  # timestamps present in the dvs

        self.events = zip(data['ts'], data['x'], data['y'], data['pol'])
        self.events_x = data['x']
        self.events_y = data['y']

        self.skeletons_ts = data_skl['ts']  # timestamps from vicon

        self.prev_skl_ts = 0.0
        self.ind = 1 if self.skeletons_ts[0] == self.prev_skl_ts else 0
        self.current_skl_ts = self.skeletons_ts[self.ind]
        self.prev_event_ts = 0

        self.stop_flag = False
        self.skl_keys = [str(i) for i in range(0, 13)]
        self.skl = data_skl

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.ceil(len(self.events_ts) / self.skeletons_ts))

    def __next__(self):

        t1 = time()

        if self.stop_flag:
            raise StopIteration

        self.prev_skl_ts = self.current_skl_ts
        self.current_skl_ts = self.skeletons_ts[self.ind]

        # Extracting all relevant events in the time frame
        events_iter = np.array([])

        # print(f'self.prev: {self.prev}, self.current: {self.current}')
        # print(f'self.prev_event_ts: {self.prev_event_ts}')
        event_found = False

        for i, t in enumerate(self.events_ts[self.prev_event_ts:]):
            if self.prev_skl_ts < t <= self.current_skl_ts:

                if not event_found:
                    self.prev_event_ts = i
                event_found = True

                # events = np.array([self.events_x[i], self.events_y[i]], dtype=int).reshape(1, 2)
                events = np.zeros((1, 2), dtype=int)
                events[0, 0] = self.events_x[i]
                events[0, 1] = self.events_y[i]
                try:
                    events_iter = np.concatenate((events, events_iter), axis=0)
                except:
                    events_iter = events
            elif t > self.current_skl_ts:
                break

        # Extracting the GT skeleton
        # skl = []
        # [skl.append(self.skl[k][self.ind]) for k in self.skl_keys]
        # skl = np.vstack(skl)
        skl = np.zeros((13, 2), dtype=int)
        for i, k in enumerate(self.skl_keys):
            skl[i] = self.skl[k][self.ind]
        self.ind += 1
        self.__update_current_index(self.ind)
        # print(events_iter.shape)

        # print(f'elapsed time for {self.__class__.__name__}.__next__: {time() - t1}')

        return events_iter, skl, self.current_skl_ts

    def __update_current_index(self, end_ind):

        if end_ind >= self.skeletons_ts.shape[0]:
            self.stop_flag = True

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
