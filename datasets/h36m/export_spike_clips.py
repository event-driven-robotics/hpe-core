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
import pathlib
import os
import h5py
import numpy as np

from datasets.utils.parsing import import_yarp_skeleton_data, timedBatchIterator
from datasets.utils.events_representation import eventFrame
from datasets.utils.export import ensure_location, str2bool, get_movenet_keypoints, get_center
from bimvee.importIitYarp import importIitYarp as import_dvs


def export_stencil_spikes(data_dvs_file, data_vicon_file, output_path, args=None, output_json=None):
    action_name = data_dvs_file.split(os.sep)[-2]
    data_vicon = import_yarp_skeleton_data(pathlib.Path(data_vicon_file))
    data_dvs = import_dvs(filePathOrName=data_dvs_file)
    data_dvs['data']['left']['dvs']['ts'] /= args.ts_scaler
    iterator = timedBatchIterator(data_dvs['data']['left']['dvs'], data_vicon, duration=.040)
    if args.dev:
        event_frame = eventFrame(frame_height=args.frame_height, frame_width=args.frame_width)

    poses_movenet = []

    for fi, (events, poses, batch_size) in enumerate(iterator):
        # for fi, (events, pose, ts) in enumerate(iterator):
        if args.dev:
            print('frame: ', fi)

        sample_anno = {}
        sample_anno['file_name'] = action_name + '_' + str(fi).zfill(5) + '.jpg'
        sample_anno['ts'] = []
        sample_anno['keypoints'] = []
        sample_anno['center'] = []
        sample_anno['image_size'] = [args.frame_width, args.frame_height]
        for pose in poses:
            sample_anno['ts'].extend([pose['ts']])
            sample_anno['keypoints'].extend(get_movenet_keypoints(pose, args.frame_height, args.frame_width))
            sample_anno['center'].extend(get_center(pose, args.frame_height, args.frame_width))
        # sample_anno['torso_size'] = get_torso_length(pose, frame_height, frame_width)
        sample_anno['original_sample'] = action_name
        if len(sample_anno['center']) == 0 or sample_anno['ts'] == [0.0] or len(events['ts'])==0:
            continue
        print(sample_anno)
        if args.dev:
            event_frame.reset_frame()
            for ei in range(batch_size):
                event_frame.update(vx=int(events['x'][ei]), vy=int(events['y'][ei]))
            keypoints = np.reshape(sample_anno['keypoints'][:39], [-1, 3])
            frame = event_frame.get_normed()
            h, w = frame.shape
            for i in range(len(keypoints)):
                frame = cv2.circle(frame, [int(keypoints[i, 0] * w), int(keypoints[i, 1] * h)], 1, (255, 0, 0), 2)
            frame = cv2.circle(frame, [int(sample_anno['center'][0] * w), int(sample_anno['center'][1] * h)], 1,
                               (255, 0, 0), 4)
            cv2.imshow('', frame)
            cv2.waitKey(1)
        poses_movenet.append(sample_anno)

        # Writing the GT
        if args.write_format is not None:
            dvs_file_name = os.path.join(output_path, str(fi).zfill(5) + args.write_format)
            sample_anno['file_name'] = dvs_file_name
            hf_write = h5py.File(dvs_file_name, 'w')
            data = hf_write.create_dataset("events", (len(events['ts']), 4))
            # try:
            data[:, 0] = (events['ts'][:]) / 1e-6
            # except IndexError:
            #     continue
            data[:, 1] = events['x'][:]
            data[:, 2] = events['y'][:]
            data[:, 3] = events['pol'][:]
            hf_write.close()

    if args.write_annotation and output_json is not None:
        if args.from_scratch:
            with open(str(output_json), 'w') as f:
                json.dump(poses_movenet, f, ensure_ascii=False)
                args.from_scratch = False
        else:
            with open(str(output_json), 'r+') as f:
                poses = json.load(f)
                poses.extend(poses_movenet)
                f.seek(0)
                json.dump(poses, f, ensure_ascii=False)
    return


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
        output_path_images = args.data_home + "/tester/h36m_stencil/"
        output_path_anno = args.data_home + "/tester/stencil_anno/"
    else:
        output_path_images = args.data_home + "/training/h36m_stencil/"
        output_path_anno = args.data_home + "/training/stencil_anno/"

    output_path_images = os.path.abspath(output_path_images)
    output_path_anno = os.path.abspath(output_path_anno)
    output_json = output_path_anno + '/poses.json'
    ensure_location(output_path_images)
    ensure_location(output_path_anno)

    input_data_dir = args.data_home + "/yarp/"
    input_data_dir = os.path.abspath(input_data_dir)

    dir_list = os.listdir(input_data_dir)
    already_done = setup_testing_list(output_json)
    if args.dev:
        dir_list = ['cam2_S11_Directions']

    for i in tqdm(range(len(dir_list))):
        sample = dir_list[i]
        cam = sample[3]
        dvs_dir = os.path.join(input_data_dir, sample, 'ch0dvs')
        data_vicon_file = os.path.join(input_data_dir, sample, f'ch{cam}GT50Hzskeleton/data.log')
        process = True
        print(str(i), "=====", sample, "=====")

        if sample in already_done:
            print('skipping: ', sample, '(already processed)')
            process = False
        elif not os.path.exists(dvs_dir):
            print('skipping: ', dvs_dir, 'does not exist')
            process = False
        elif not os.path.exists(data_vicon_file):
            print('skipping: ', data_vicon_file, 'does not exist')
            process = False

        if process:
            export_stencil_spikes(dvs_dir, data_vicon_file, output_path_images, args=args, output_json=output_json)

        if args.dev:
            exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-frame_width', help='', default=640, type=int)
    parser.add_argument('-frame_height', help='', default=480, type=int)
    parser.add_argument('-data_home', help='Path to dataset folder', default='/home/ggoyal/data/h36m_cropped/',
                        type=str)
    parser.add_argument("-from_scratch", type=str2bool, nargs='?', const=True, default=True,
                        help="Write annotation file from scratch.")
    parser.add_argument("-write_annotation", type=str2bool, nargs='?', const=True, default=True,
                        help="Write annotation file.")
    parser.add_argument("-write_format", type=str, default='.h5', help="None, .h5, .data")
    parser.add_argument('-ms', help='Duration of the clips', default=40, type=int)
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=True, help="Run in dev mode.")
    parser.add_argument("-ts_scaler", help='', default=12.50, type=float)

    args = parser.parse_args()
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print('Catching an argumentError')
    main(args)
