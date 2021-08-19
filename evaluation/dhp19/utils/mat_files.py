
import numpy as np
import scipy.io as spio


# DVS camera
DHP19_SENSOR_HEIGHT = 260
DHP19_SENSOR_WIDTH = 346


class Dhp19EventsIterator:

    def __init__(self, data, cam_id, window_size):
        self.events = data['out']['data'][f'cam{cam_id}']['dvs']

        # events location indices follow matlab indexing convention, i.e. they start from 1 instead of 0
        self.events['x'] = self.events['x'] - 1
        self.events['y'] = self.events['y'] - 1

        # events x indices are shifted of sensor_width * camera id
        self.events['x'] = self.events['x'] - DHP19_SENSOR_WIDTH * cam_id

        self.window_size = window_size
        self.curr_ind = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_ind == -1:
            raise StopIteration

        end_ind = self.curr_ind + self.window_size
        data = np.concatenate((np.reshape(self.events['ts'][self.curr_ind:end_ind], (-1, 1)),
                               np.reshape(self.events['x'][self.curr_ind:end_ind], (-1, 1)),
                               np.reshape(self.events['y'][self.curr_ind:end_ind], (-1, 1)),
                               np.reshape(self.events['pol'][self.curr_ind:end_ind], (-1, 1))), axis=1, dtype=np.float64)

        if end_ind > self.events['x'].shape[0]:
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
