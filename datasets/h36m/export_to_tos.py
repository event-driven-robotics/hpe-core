"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author:
    Gaurvi Goyal

LICENSE GOES HERE
"""

import argparse
import cv2
import pathlib
import json
import bimvee
from datasets.utils.events_representation import EROS

data_dvs_file = "/home/ggoyal/data/h36m/yarp/S1_Directions/ch0dvs/data.log"
data_vicon_file = "/home/ggoyal/data/h36m/yarp/S1_Directions/ch0GT50Hzskeleton/data.log"
output_path = "/home/ggoyal/data/h36m/tester/yarp/S1_Directions/"

with open(data_dvs_file) as f:
    data_dvs = f.readlines()

with open(data_vicon_file) as f:
    data_vicon = f.readlines()

print(len(data_dvs))
print(len(data_vicon))

# Set up the iterator. where the next function will call all events till the next timestamp of the skeleton

# Loop on iterator.
    # update the EROS represetnation and save the image in jpg.
    # Save the pose information for the json file.
# Shuffle the pose files. Save the json file.

def importSkeletonDataLog(filePath, keys):
    # TODO: Read and safe skeletons in the movenet skeleton points.
    pass
    # pattern = re.compile('\d* (\d*.\d*) SKLT \((.*)\)')
    # with open(filePath) as f:
    #     content = f.readlines()
    # data_dict = {k: [] for k in keys}
    # timestamps = []
    # for line in content:
    #     ts, data = pattern.findall(line)[0]
    #     data = np.array(data.split(' ')).astype(int).reshape(-1, 2)
    #     for d, label in zip(data, data_dict):
    #         data_dict[label].append(d)
    #     timestamps.append(ts)
    # data_dict['ts'] = np.array(timestamps).astype(float)
    # for d in data_dict:
    #     data_dict[d] = np.array(data_dict[d])
    # return data_dict

# def importSkeleton(**kwargs):
#     with open(kwargs.get('filePathOrName'), 'r') as f:
#         data_dict = json.load(f)
#     return importSkeletonDataLog(os.path.join(os.path.dirname(kwargs['filePathOrName']), data_dict['file']), data_dict['labels'])

def importeventDataLog(filePath, keys):
    # TODO: Read and safe dvs in the format that the EROS needs it.
    pass

class H36M19Iterator:

    def __init__(self, data_dvs, data_vicon):
        self.data_dvs = data_dvs
        self.data_vicon = data_vicon


    def __iter__(self):
        return self

    def __len__(self):
        return 0 #TODO: Number of poses.

    # def __next__(self):
#         if self.curr_ind == -1:
#             raise StopIteration
#
#         end_ind = self.curr_ind + self.window_size
#
#         # select events from the specified camera with timestamps within the current window
#         window_timestamps = self.timestamps[self.curr_ind:end_ind]
#         event_indices = np.isin(self.events['ts'], window_timestamps)
#         window_events = np.concatenate((np.reshape(self.events['ts'][event_indices], (-1, 1)),
#                                         np.reshape(self.events['x'][event_indices], (-1, 1)),
#                                         np.reshape(self.events['y'][event_indices], (-1, 1)),
#                                         np.reshape(self.events['pol'][event_indices], (-1, 1))),
#                                        axis=1, dtype=np.float64)
#
#         self.__update_current_index(end_ind)
#
#         if self.vicon is None:
#             return window_events
#
#         # get indices of the 3d poses inside the window
#         # poses_start_ind = int(np.floor((window_timestamps[0] - self.start_time) * 1e-4))
#         # poses_end_ind = int(np.floor((window_timestamps[-1] - self.start_time) * 1e-4))
#         poses_start_ind = window_timestamps[0]
#         poses_end_ind = window_timestamps[-1]
#
#         # compute the average 3d pose
#         avg_pose_3d = np.zeros(shape=(len(DHP19_BODY_PARTS), 3))
#         for body_part in DHP19_BODY_PARTS:
#             coords = self.vicon[body_part][poses_start_ind:poses_end_ind, :]
#             avg_pose_3d[DHP19_BODY_PARTS[body_part], :] = np.nanmean(coords, axis=0)
#
#         if self.proj_mat is None:
#             return window_events, avg_pose_3d
#
#         # project the 3d pose to the camera plane
#         avg_pose_2d, joints_mask = project_poses_to_2d(avg_pose_3d[np.newaxis, :, :], np.transpose(self.proj_mat))
#
#         return window_events, avg_pose_2d
#
#     def __update_current_index(self, end_ind):
#
#         if end_ind >= self.timestamps.shape[0]:
#             self.curr_ind = -1
#         else:
#             if self.stride:
#                 self.curr_ind += self.stride
#                 if self.curr_ind >= self.timestamps.shape[0]:
#                     self.curr_ind = -1
#             else:
#                 self.curr_ind = end_ind
