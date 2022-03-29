
import numpy as np
import pathlib
import re

from typing import Dict

from datasets.utils.constants import HPECoreSkeleton


def import_yarp_skeleton_data(yarp_file_path: pathlib.Path) -> Dict:

    with open(str(yarp_file_path.resolve())) as f:
        content = f.readlines()

    # line format is 'id timestamp SKLT (<flattened 2D keypoints coordinates>) head_size torso_size'
    pattern = re.compile('\d* (\d*.\d*) SKLT \((.*)\) (-?\d*.\d*) (\d*.\d*)')

    timestamps = np.zeros((len(content)), dtype=float)
    head_sizes = np.zeros((len(content)), dtype=float)
    torso_sizes = np.zeros((len(content)), dtype=float)
    data_dict = {k: [] for k in HPECoreSkeleton.KEYPOINTS_MAP}

    for li, line in enumerate(content):
        tss, points, head_size, torso_size = pattern.findall(line)[0]

        points = np.array(list(filter(None, points.split(' ')))).astype(int).reshape(-1, 2)
        for d, label in zip(points, HPECoreSkeleton.KEYPOINTS_MAP):
            data_dict[label].append(d)

        head_sizes[li] = head_size
        torso_sizes[li] = torso_size

        timestamps[li] = tss

    data_dict['ts'] = timestamps
    data_dict['head_sizes'] = head_sizes
    data_dict['torso_sizes'] = torso_sizes

    for d in data_dict:
        data_dict[d] = np.array(data_dict[d])
    return data_dict


class YarpHPEIterator:
    def __init__(self, data_events: Dict, data_skl: Dict):

        self.events_ts = data_events['ts']
        self.events_x = data_events['x']
        self.events_y = data_events['y']
        self.events_pol = data_events['pol']

        self.skeletons_ts = data_skl['ts']
        self.skeletons = data_skl
        self.skeleton_keys = [str(i) for i in range(len(HPECoreSkeleton.KEYPOINTS_MAP))]

        self.prev_skl_ts = 0.0
        self.ind = 1 if self.skeletons_ts[0] == self.prev_skl_ts else 0
        self.current_skl_ts = self.skeletons_ts[self.ind]
        self.prev_event_ts = 0

        self.stop_flag = False

    def __iter__(self):
        return self

    def __len__(self):
        return self.skeletons_ts.shape[0]

    def __next__(self):

        if self.stop_flag:
            raise StopIteration

        self.prev_skl_ts = self.current_skl_ts
        self.current_skl_ts = self.skeletons_ts[self.ind]

        # select events between the previous and the current poses
        event_indices = (self.prev_skl_ts < self.events_ts) & (self.events_ts <= self.current_skl_ts)
        window_events = np.concatenate((np.reshape(self.events_ts[event_indices], (-1, 1)),
                                        np.reshape(self.events_x[event_indices], (-1, 1)),
                                        np.reshape(self.events_y[event_indices], (-1, 1)),
                                        np.reshape(self.events_pol[event_indices], (-1, 1))),
                                       axis=1, dtype=np.float64)

        # extract ground truth pose
        skl = np.zeros((len(HPECoreSkeleton.KEYPOINTS_MAP), 2), dtype=int)
        for i, k in enumerate(self.skeleton_keys):
            skl[i] = self.skeletons[k][self.ind]

        self.__update_current_index()

        return window_events, skl, self.current_skl_ts

    def __update_current_index(self):

        self.ind += 1
        if self.ind >= self.skeletons_ts.shape[0]:
            self.stop_flag = True
