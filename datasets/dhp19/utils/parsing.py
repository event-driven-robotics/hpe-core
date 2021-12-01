
import numpy as np

import constants as dhp19_const

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

OPENPOSE_BODY_25_TO_DHP19_INDICES = np.array([
    # TODO: compute head
    [0, DHP19_BODY_PARTS['head']],
    [2, DHP19_BODY_PARTS['shoulderR']],
    [5, DHP19_BODY_PARTS['shoulderL']],
    [3, DHP19_BODY_PARTS['elbowR']],
    [6, DHP19_BODY_PARTS['elbowL']],
    [12, DHP19_BODY_PARTS['hipL']],
    [9, DHP19_BODY_PARTS['hipR']],
    [4, DHP19_BODY_PARTS['handR']],
    [7, DHP19_BODY_PARTS['handL']],
    [10, DHP19_BODY_PARTS['kneeR']],
    [13, DHP19_BODY_PARTS['kneeL']],
    [11, DHP19_BODY_PARTS['footR']],
    [14, DHP19_BODY_PARTS['footL']]
])


def openpose_to_dhp19(pose_op):
    # TODO: compute dhp19's head joints from openpose
    return pose_op[OPENPOSE_BODY_25_TO_DHP19_INDICES[:, 0], :]


class Dhp19EventsIterator:

    # TODO: add param for overlapping?
    # def __init__(self, data, cam_id, window_size=DHP19_FRAME_EVENTS_NUM, stride=None):
    def __init__(self, data, cam_id, window_size=dhp19_const.DHP19_CAM_FRAME_EVENTS_NUM):

        self.timestamps = data['out']['extra']['ts']  # array containing timestamps of events from all cameras

        self.events = data['out']['data'][f'cam{cam_id}']['dvs']  # events specific to selected camera

        # events location indices follow matlab indexing convention, i.e. they start from 1 instead of 0
        self.events['x'] = self.events['x'] - 1
        self.events['y'] = self.events['y'] - 1

        # events x indices are shifted by sensor_width * camera id
        self.events['x'] = self.events['x'] - dhp19_const.DHP19_SENSOR_WIDTH * cam_id

        # events are sampled from all cameras, thus the actual window size is the desired input one (representing the
        # desired number of frames from a single camera) multiplied by the number of cameras (for an explanation, see
        # Section 4.1 of paper "DHP19: Dynamic Vision Sensor 3D Human Pose Dataset")
        self.window_size = window_size * dhp19_const.DHP19_CAM_NUM

        self.curr_ind = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.ceil(len(self.timestamps) / self.window_size))

    def __next__(self):
        if self.curr_ind == -1:
            raise StopIteration

        end_ind = self.curr_ind + self.window_size

        # select events from the specified camera with timestamps within the current window
        window_timestamps = self.timestamps[self.curr_ind:end_ind]
        event_indices = np.isin(self.events['ts'], window_timestamps)
        data = np.concatenate((np.reshape(self.events['ts'][event_indices], (-1, 1)),
                               np.reshape(self.events['x'][event_indices], (-1, 1)),
                               np.reshape(self.events['y'][event_indices], (-1, 1)),
                               np.reshape(self.events['pol'][event_indices], (-1, 1))), axis=1, dtype=np.float64)

        if end_ind >= self.timestamps.shape[0]:
            self.curr_ind = -1
        else:
            self.curr_ind = end_ind

        return data
