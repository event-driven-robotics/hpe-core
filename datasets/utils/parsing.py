
import numpy as np
import pathlib
import re

from typing import Dict

from datasets.utils.constants import HPECoreSkeleton


def import_yarp_skeleton_data(yarp_file_path: pathlib.Path) -> Dict:

    with open(str(yarp_file_path.resolve())) as f:
        content = f.readlines()

    if(len(content) == 0):
        raise Exception("No file, or no file content") 

    # line format is 'id timestamp SKLT (<flattened 2D keypoints coordinates>) head_size torso_size'
    pattern = re.compile('\d* (\d*.\d*) SKLT \((.*)\) (-?\d*.\d*) (\d*.\d*)')

    timestamps = np.zeros((len(content)), dtype=float)
    head_sizes = np.zeros((len(content)), dtype=float)
    torso_sizes = np.zeros((len(content)), dtype=float)
    data_dict = {k: [] for k in HPECoreSkeleton.KEYPOINTS_MAP}

    try:
        tss, points, head_size, torso_size = pattern.findall(content[0])[0]
    except:
        print("Dataset", yarp_file_path, "does not match pattern")
        print("required: [# TS SKLT (int x26) head torso]")
        print("got     :", content[0])
        exit()

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

class batchIterator:
    def __init__(self, events: Dict, gt_skel: Dict, n = 0, offset = 0):

        self._index = 0
        self._events = events
        self._gt_skel = gt_skel
        self._samples_count = len(gt_skel['ts'])
        self._events_count = len(events['ts'])
        self._skel_sample = [[0,0]] * len(gt_skel)
        self.n = n
        self.offset= offset

        self._batch_indices = [0] * (self._samples_count+1)

        gt_i = 0
        ev_i = 0
        for ev_i in range(self._events_count):
            if self._events['ts'][ev_i] > gt_skel['ts'][gt_i]:
                self._batch_indices[gt_i+1] = ev_i
                gt_i += 1
                if(gt_i >= self._samples_count):
                    break

    def __iter__(self):
        return self

    def __len__(self):
        return self._samples_count

    def __next__(self):

        if self._index >= self._samples_count: 
            raise StopIteration

        i1 = self._batch_indices[self._index]
        if self.n !=0:
            i1 = self._batch_indices[self._index+1]-self.n +self.offset
        i2 = self._batch_indices[self._index+1]+self.offset

        retv = dict()
        retv['ts'] = self._events['ts'][i1:i2]
        retv['x'] = self._events['x'][i1:i2]
        retv['y'] = self._events['y'][i1:i2]
        retv['pol'] = self._events['pol'][i1:i2]

        rets = dict()
        for jname in self._gt_skel.keys():
            rets[jname] = self._gt_skel[jname][self._index]

        self._index += 1
        return retv, rets, len(retv['ts'])

class timedBatchIterator:
    def __init__(self, events: Dict, gt_skel: Dict, duration=.040):
        # Duration: time in ms
        self._index = 0
        self._events = events
        self._gt_skel = gt_skel
        self._samples_count = len(gt_skel['ts'])
        self._events_count = len(events['ts'])
        self._skel_sample = [[0,0]] * len(gt_skel)
        self.duration = duration
        total_time = events['ts'][-1]-events['ts'][0]
        self._batch_indices = [0] * (int(total_time/duration) +1)
        self._skeleton_indices = [0] * (int(total_time/duration) +1)

        gt_i = 0
        ev_i = 0
        ts = events['ts'][0]
        count = 0
        for ev_i in range(self._events_count):
            if self._gt_skel['ts'][gt_i]<self._events['ts'][ev_i]:
                self._skeleton_indices[count+1] = gt_i+1
                gt_i += 1
                if (gt_i +1 >= self._samples_count):
                    break
            if self._events['ts'][ev_i] > ts+duration:
                self._batch_indices[count+1] = ev_i
                ts = self._events['ts'][ev_i]
                count +=1
                if(gt_i >= self._samples_count):
                    break

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._skeleton_indices-1)

    def __next__(self):

        if self._index + 1 >= len(self._batch_indices):
            raise StopIteration

        i1 = self._batch_indices[self._index]
        i2 = self._batch_indices[self._index+1]
        j1 = self._skeleton_indices[self._index]
        j2 = self._skeleton_indices[self._index+1]

        retv = dict()
        retv['ts'] = self._events['ts'][i1:i2]
        retv['x'] = self._events['x'][i1:i2]
        retv['y'] = self._events['y'][i1:i2]
        retv['pol'] = self._events['pol'][i1:i2]

        rets = []
        for i in range(j1,j2+1):
            rets_i = dict()
            for jname in self._gt_skel.keys():
                rets_i[jname] = self._gt_skel[jname][i]
                rets_i['ts'] = self._gt_skel['ts'][i]
            rets.append(rets_i)

        self._index += 1
        return retv, rets, len(retv['ts'])


class YarpHPEIterator:
    def __init__(self, data_events: Dict, data_skl: Dict):

        self.events_ts = data_events['ts']
        self.events_x = data_events['x']
        self.events_y = data_events['y']
        self.events_pol = data_events['pol']

        self.skeletons_ts = data_skl['ts']
        self.skeletons = data_skl
        self.skeleton_keys = [str(i) for i in range(len(HPECoreSkeleton.KEYPOINTS_MAP))]

        self.head_sizes = data_skl['head_sizes']
        self.torso_sizes = data_skl['torso_sizes']

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
        for i, k in enumerate(HPECoreSkeleton.KEYPOINTS_MAP):
            skl[i] = self.skeletons[k][self.ind]

        self.__update_current_index()

        return window_events, skl, self.current_skl_ts, self.head_sizes[self.ind], self.torso_sizes[self.ind]

    def __update_current_index(self):

        self.ind += 1
        if self.ind >= self.skeletons_ts.shape[0]:
            self.stop_flag = True
