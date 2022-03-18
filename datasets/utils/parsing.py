
import numpy as np
import pathlib
import re

from typing import Dict

from datasets.utils.constants import HPECoreSkeleton


def import_skeleton_data(yarp_file_path: pathlib.Path) -> Dict:

    with open(str(yarp_file_path.resolve())) as f:
        content = f.readlines()

    pattern = re.compile('\d* (\d*.\d*) SKLT \((.*)\)')

    timestamps = np.zeros((len(content)), dtype=float)
    data_dict = {k: [] for k in HPECoreSkeleton.KEYPOINT_LABELS}
    for li, line in enumerate(content):
        tss, points = pattern.findall(line)[0]
        points = np.array(list(filter(None, points.split(' ')))).astype(int).reshape(-1, 2)
        for d, label in zip(points, HPECoreSkeleton.KEYPOINT_LABELS):
            data_dict[label].append(d)
        timestamps[li] = tss
    data_dict['ts'] = timestamps
    for d in data_dict:
        data_dict[d] = np.array(data_dict[d])
    return data_dict


class YarpHPEIterator:
    def __init__(self, data_events, data_skl):

        self.events_ts = data_events['ts']  # timestamps present in the dvs

        self.events = zip(data_events['ts'], data_events['x'], data_events['y'], data_events['pol'])
        self.events_x = data_events['x']
        self.events_y = data_events['y']

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

        if self.stop_flag:
            raise StopIteration

        self.prev_skl_ts = self.current_skl_ts
        self.current_skl_ts = self.skeletons_ts[self.ind]

        # Extracting all relevant events in the time frame
        events_iter = np.array([])

        event_found = False

        for i, t in enumerate(self.events_ts[self.prev_event_ts:]):
            if self.prev_skl_ts < t <= self.current_skl_ts:

                if not event_found:
                    self.prev_event_ts = i
                event_found = True

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
        skl = np.zeros((13, 2), dtype=int)
        for i, k in enumerate(self.skl_keys):
            skl[i] = self.skl[k][self.ind]

        self.__update_current_index()

        return events_iter, skl, self.current_skl_ts

    def __update_current_index(self):

        self.ind += 1
        if self.ind >= self.skeletons_ts.shape[0]:
            self.stop_flag = True
