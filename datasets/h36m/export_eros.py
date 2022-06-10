"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author:
    Gaurvi Goyal

LICENSE GOES HERE
"""

from tqdm import tqdm
import argparse
import cv2
import json
import os
import re
import math
import numpy as np
from datasets.h36m.utils.parsing import H36mIterator, hpecore_to_movenet
from datasets.utils.events_representation import EROS
from datasets.utils.export import ensure_location
from bimvee.importIitYarp import importIitYarp as import_dvs




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        points_movenet = hpecore_to_movenet(points)
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
    # x_cen = np.mean([min(pose[:, 0]), max(pose[:, 0])]) / w_frame
    # y_cen = np.mean([min(pose[:, 1]), max(pose[:, 1])]) / h_frame
    x_cen = np.mean(pose[:, 0]) / w_frame
    y_cen = np.mean(pose[:, 1]) / h_frame
    return [x_cen, y_cen]


def export_to_eros(data_dvs_file, data_vicon_file, output_path_images, skip=None, args=None):
    if skip == None:
        skip = 1
    else:
        skip = int(skip) + 1
    action_name = data_dvs_file.split(os.sep)[-2]

    data_vicon = importSkeletonData(data_vicon_file)
    data_dvs = import_dvs(filePathOrName=data_dvs_file)

    iterator = H36mIterator(data_dvs['data']['left']['dvs'], data_vicon,time_factor=12.5)
    eros = EROS(kernel_size=args.eros_kernel, frame_width=args.frame_width, frame_height=args.frame_height)

    poses_movenet = []
    for fi, (events, pose, ts) in enumerate(iterator):
        if args.dev:
            print('frame: ', fi)
        for ei in range(len(events)):
            eros.update(vx=int(events[ei, 0]), vy=int(events[ei, 1]))
        frame = eros.get_frame()

        if fi < 2:  # Almost empty images, not beneficial for training
            kps_old = get_keypoints(pose, args.frame_height, args.frame_width)
            continue

        if fi % skip == 0:
            sample_anno = {}
            sample_anno['img_name'] = action_name + '_' + str(fi) + '.jpg'
            sample_anno['ts'] = ts
            sample_anno['keypoints'] = get_keypoints(pose, args.frame_height, args.frame_width)
            sample_anno['center'] = get_center(pose, args.frame_height, args.frame_width)
            # sample_anno['torso_size'] = get_torso_length(pose, frame_height, frame_width)
            sample_anno['keypoints_prev'] = kps_old
            sample_anno['original_sample'] = action_name

            # print(sample_anno)
            frame = cv2.GaussianBlur(frame, (args.gauss_kernel, args.gauss_kernel), 0)
            if args.dev:
                keypoints = np.reshape(sample_anno['keypoints'], [-1, 3])
                h, w = frame.shape
                for i in range(len(keypoints)):
                    frame = cv2.circle(frame, [int(keypoints[i, 0] * w), int(keypoints[i, 1] * h)], 1, (255, 0, 0), 2)
                frame = cv2.circle(frame, [int(sample_anno['center'][0] * w), int(sample_anno['center'][1] * h)], 1,
                                   (255, 0, 0), 4)
                cv2.imshow('', frame)
                cv2.waitKey(1)
            else:
                cv2.imwrite(os.path.join(output_path_images, sample_anno['img_name']), frame)
            poses_movenet.append(sample_anno)

            kps_old = sample_anno['keypoints']
    return poses_movenet


def setup_testing_list(path):
    if not os.path.exists(path):
        return []
    with open(str(path), 'r+') as f:
        poses = json.load(f)
    files = [sample['original_sample'] for sample in poses]
    files_unique = set(files)
    return files_unique


def main(args):
    if args.dev:
        output_path_images = args.data_home + "/tester/h36m_EROS/"
        output_path_anno = args.data_home + "/tester/h36m_anno/"
    else:
        output_path_images = args.data_home + "/training/h36m_EROS/"
        output_path_anno = args.data_home + "/training/h36m_anno/"

    output_path_images = os.path.abspath(output_path_images)
    output_path_anno = os.path.abspath(output_path_anno)
    output_json = output_path_anno + '/poses.json'
    ensure_location(output_path_images)
    ensure_location(output_path_anno)

    input_data_dir = args.data_home + "/../h36m_cropped/yarp/"
    input_data_dir = os.path.abspath(input_data_dir)

    dir_list = os.listdir(input_data_dir)

    already_done = setup_testing_list(output_json)
    # print(already_done)
    for i in tqdm(range(len(dir_list))):
        if args.dev:
            sample = 'cam2_S5_Directions_1'
        else:
            sample = dir_list[i]
        cam = sample[3]
        dvs_dir = os.path.join(input_data_dir, sample, 'ch0dvs')
        data_vicon_file = os.path.join(input_data_dir, sample, f'ch{cam}GT50Hzskeleton/data.log')
        print(str(i), sample)
        print(data_vicon_file)
        if sample in already_done:
            continue

        if os.path.exists(dvs_dir) and os.path.exists(data_vicon_file):
            poses_sample = export_to_eros(dvs_dir, data_vicon_file, output_path_images, skip=args.skip_image, args=args)

            if args.write_annotation:
                if args.from_scratch:
                    with open(str(output_json), 'w') as f:
                        json.dump(poses_sample, f, ensure_ascii=False)
                        args.from_scratch = False
                        if args.dev:
                            exit()
                else:
                    with open(str(output_json), 'r+') as f:
                        poses = json.load(f)
                        poses.extend(poses_sample)
                        f.seek(0)
                        json.dump(poses, f, ensure_ascii=False)

        else:
            print(f"File {sample} not found.")
            if args.dev:
                print(os.path.exists(dvs_dir))
                print(os.path.exists(data_vicon_file))

        if args.dev:
            exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-eros_kernel', help='', default=6, type=int)
    parser.add_argument('-frame_width', help='', default=640, type=int)
    parser.add_argument('-frame_height', help='', default=480, type=int)
    parser.add_argument('-gauss_kernel', help='', default=7, type=int)
    parser.add_argument('-skip_image', help='', default=None)
    parser.add_argument('-data_home', help='Path to dataset folder', default='/home/ggoyal/data/h36m/', type=str)
    parser.add_argument("-from_scratch", type=str2bool, nargs='?', const=True, default=True,
                        help="Write annotation file from scratch.")
    parser.add_argument("-write_annotation", type=str2bool, nargs='?', const=True, default=True,
                        help="Write annotation file.")
    parser.add_argument("-write_images", type=str2bool, nargs='?', const=True, default=True, help="Save images.")
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=False, help="Run in dev mode.")

    args = parser.parse_args()
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print('Catching an argumentError')
    main(args)
    # eros_kernel = 6
    # frame_width = 640
    # frame_height = 480
    # gauss_kernel = 7
    # data_home = '/home/ggoyal/data/h36m/'
    # from_scratch = True
    # write_annotation = True
    # write_images = True
    # dev = True
    # skip_image = 4

