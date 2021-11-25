
import numpy as np


# DVS camera
DHP19_SENSOR_HEIGHT = 260
DHP19_SENSOR_WIDTH = 346
DHP19_CAM_FRAME_EVENTS_NUM = 7500  # fixed number of events used in DHP19 for generating event frames
DHP19_CAM_NUM = 4  # number of synchronized cameras used for recording events

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


def openpose_to_dhp19(pose_op):
    # TODO: compute dhp19's head joints from openpose
    return pose_op[OPENPOSE_BODY_25_TO_DHP19_INDICES[:, 0], :]


class Dhp19EventsIterator:

    # TODO: add param for overlapping?
    # def __init__(self, data, cam_id, window_size=DHP19_FRAME_EVENTS_NUM, stride=None):
    def __init__(self, data, cam_id, window_size=DHP19_CAM_FRAME_EVENTS_NUM):

        self.timestamps = data['out']['extra']['ts']  # array containing timestamps of events from all cameras

        self.events = data['out']['data'][f'cam{cam_id}']['dvs']  # events specific to selected camera

        # events location indices follow matlab indexing convention, i.e. they start from 1 instead of 0
        self.events['x'] = self.events['x'] - 1
        self.events['y'] = self.events['y'] - 1

        # events x indices are shifted by sensor_width * camera id
        self.events['x'] = self.events['x'] - DHP19_SENSOR_WIDTH * cam_id

        # events are sampled from all cameras, thus the actual window size is the desired input one (representing the
        # desired number of frames from a single camera) multiplied by the number of cameras (for an explanation, see
        # Section 4.1 of paper "DHP19: Dynamic Vision Sensor 3D Human Pose Dataset")
        self.window_size = window_size * DHP19_CAM_NUM

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
