
import numpy as np
import scipy.io as spio


# DVS camera
DHP19_SENSOR_HEIGHT = 260
DHP19_SENSOR_WIDTH = 346
DHP19_CAM_FRAME_EVENTS_NUM = 7500  # fixed number of events used in DHP19 for generating event frames
DHP19_CAM_NUM = 4  # number of synchronized cameras used for recording events


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


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


###########
# PRIVATE #
###########

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
