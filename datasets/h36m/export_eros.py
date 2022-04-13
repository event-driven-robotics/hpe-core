"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author:
    Gaurvi Goyal

LICENSE GOES HERE
"""

from tqdm import tqdm
import cv2
import json
import os
import re
import math
import numpy as np
from datasets.h36m.utils.parsing import H36mIterator, h36m_to_movenet
from datasets.utils.events_representation import EROS
from bimvee.importIitYarp import importIitYarp as import_dvs

eros_kernel = 6
frame_width = 346
frame_height = 260
gauss_kernel = 7


def importSkeletonData(filename):
    with open(filename) as f:
        content = f.readlines()

    # TODO: Read and safe skeletons in the movenet skeleton points.
    pass
    pattern = re.compile('\d* (\d*.\d*) SKLT \((.*)\)')

    data_dict = {}
    timestamps = []
    keys = [str(i) for i in range(0, 13)]
    data_dict = {k: [] for k in keys}
    for line in content:
        tss, points = pattern.findall(line)[0]
        points = np.array(list(filter(None, points.split(' ')))).astype(int).reshape(-1, 2)
        # points = np.array(points.split(' ')).astype(int).reshape(-1, 2)
        points_movenet = h36m_to_movenet(points)
        for d, label in zip(points_movenet, keys):
            data_dict[label].append(d)
        timestamps.append(tss)
    data_dict['ts'] = np.array(timestamps).astype(float)
    for d in data_dict:
        data_dict[d] = np.array(data_dict[d])
    return data_dict


def get_keypoints(pose, h_frame=1, w_frame=1, add_visibility=True):
    keypoints = []
    for k in pose:
        if add_visibility:
            k_scaled = [k[0] / w_frame, k[1] / h_frame, 2]
        else:
            k_scaled = [k[0] / w_frame, k[1] / h_frame]
        keypoints.extend(k_scaled)
    return keypoints


def get_torso_length(pose, h_frame=1, w_frame=1):
    k = {}
    # k['left_shoulder']=pose[1,:]
    # k['right_shoulder']=pose[2,:]
    # k['left_hip']=pose[7,:]
    # k['right_hip']=pose[8,:]

    k['shoulder_mean'] = np.mean(pose[1:3, :], axis=0)
    k['hip_mean'] = np.mean(pose[7:9, :], axis=0)
    k['shoulder_mean'] = k['shoulder_mean'][0] / w_frame, k['shoulder_mean'][1] / h_frame
    k['hip_mean'] = k['hip_mean'][0] / w_frame, k['hip_mean'][1] / h_frame
    k['torso_dist'] = math.dist(k['shoulder_mean'], k['hip_mean'])

    return k['torso_dist']


def get_center(pose, h_frame=1, w_frame=1):
    x_cen = np.mean([min(pose[:, 0]), max(pose[:, 0])]) / w_frame
    y_cen = np.mean([min(pose[:, 1]), max(pose[:, 1])]) / h_frame
    return [x_cen, y_cen]


def export_to_eros(data_dvs_file, data_vicon_file, output_path_images):
    action_name = data_dvs_file.split(os.sep)[-2]

    data_vicon = importSkeletonData(data_vicon_file)
    data_dvs = import_dvs(filePathOrName=data_dvs_file)

    iterator = H36mIterator(data_dvs['data']['left']['dvs'], data_vicon)
    # eros = EROS(kernel_size=eros_kernel, frame_width=frame_width, frame_height=frame_height)

    poses_movenet = []
    for fi, (events, pose, ts) in enumerate(iterator):
        # for ei in range(len(events)):
        #     eros.update(vx=int(events[ei, 0]), vy=int(events[ei, 1]))
        # frame = eros.get_frame()

        if fi == 0:  # Almost empty image, not beneficial for training
            kps_old = get_keypoints(pose, frame_height, frame_width)
            continue

        sample_anno = {}
        sample_anno['img_name'] = action_name + '_' + str(fi) + '.jpg'
        sample_anno['ts'] = ts
        sample_anno['keypoints'] = get_keypoints(pose, frame_height, frame_width)
        sample_anno['center'] = get_center(pose, frame_height, frame_width)
        sample_anno['torso_size'] = get_torso_length(pose, frame_height, frame_width)
        sample_anno['keypoints_prev'] = kps_old
        sample_anno['original_sample'] = action_name

        # print(sample_anno)
        # frame = cv2.GaussianBlur(frame, (gauss_kernel, gauss_kernel), 0)
        # cv2.imwrite(os.path.join(output_path_images, sample_anno['img_name']), frame)

        poses_movenet.append(sample_anno)

        kps_old = sample_anno['keypoints']
    return poses_movenet


if __name__ == '__main__':
    from_scratch = True # Set to false if continuing a previous run

    # dvs_dir = "/home/ggoyal/data/h36m/yarp/S11_Phoning_3/ch0dvs/"
    # data_vicon_file = "/home/ggoyal/data/h36m/yarp/S11_Phoning_3/ch0GT50Hzskeleton/data.log"

    output_path_images = "/home/ggoyal/data/h36m/tester/h36m_EROS/"
    output_path_anno = "/home/ggoyal/data/h36m/tester/h36m_anno/"
    output_path_images = os.path.abspath(output_path_images)
    output_path_anno = os.path.abspath(output_path_anno)
    output_json = output_path_anno + '/poses.json'

    input_data_dir = "/home/ggoyal/data/h36m/yarp"
    input_data_dir = os.path.abspath(input_data_dir)

    dir_list = os.listdir(input_data_dir)
    print(dir_list)

    # for sample in dir_list:
    for i in tqdm(range(len(dir_list))):
        sample = dir_list[i]
        dvs_dir = os.path.join(input_data_dir, sample, 'ch0dvs')
        data_vicon_file = os.path.join(input_data_dir, sample, 'ch0GT50Hzskeleton/data.log')
        print(str(i) + sample)
        if os.path.exists(dvs_dir) and os.path.exists(data_vicon_file):
            poses_sample = export_to_eros(dvs_dir, data_vicon_file, output_path_images)

            if from_scratch:
                with open(str(output_json), 'w') as f:
                    json.dump(poses_sample, f, ensure_ascii=False)
                    from_scratch = False
                    exit() ######################################################################
            else:
                with open(str(output_json), 'r+') as f:
                    poses = json.load(f)
                    poses.extend(poses_sample)
                    f.seek(0)
                    poses = json.dump(poses, f, ensure_ascii=False)
            # poses.extend(poses_sample)



# TODO: Shuffle the pose files. Save the json file.
