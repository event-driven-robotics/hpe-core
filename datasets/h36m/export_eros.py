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
import numpy as np

from datasets.utils.parsing import import_yarp_skeleton_data, batchIterator
from datasets.utils.events_representation import EROS
from datasets.utils.export import ensure_location, str2bool, get_movenet_keypoints, get_center, get_torso_length
from bimvee.importIitYarp import importIitYarp as import_dvs


def export_to_eros(data_dvs_file, data_vicon_file, output_path, skip=None, args=None, output_json=None):
    if skip == None:
        skip = 1
    else:
        skip = int(skip) + 1
    action_name = data_dvs_file.split(os.sep)[-2]

    data_vicon = import_yarp_skeleton_data(pathlib.Path(data_vicon_file))
    data_dvs = import_dvs(filePathOrName=data_dvs_file)
    data_dvs['data']['left']['dvs']['ts'] /= args.ts_scaler
    iterator = batchIterator(data_dvs['data']['left']['dvs'], data_vicon)
    eros = EROS(kernel_size=args.eros_kernel, frame_width=args.frame_width, frame_height=args.frame_height)

    poses_movenet = []
    if args.write_video:
        output_path_video = os.path.join(output_path, action_name + '.mp4')
        print(output_path_video)
        video_out = cv2.VideoWriter(output_path_video, cv2.VideoWriter_fourcc(*'avc1'), args.fps,
                                    (args.frame_width, args.frame_height))

    for fi, (events, pose, batch_size) in enumerate(iterator):
        # for fi, (events, pose, ts) in enumerate(iterator):
        if args.dev:
            print('frame: ', fi)
        for ei in range(batch_size):
            eros.update(vx=int(events['x'][ei]), vy=int(events['y'][ei]))

        if fi < 2:  # Almost empty images, not beneficial for training
            kps_old = get_movenet_keypoints(pose, args.frame_height, args.frame_width)
            continue

        if fi % skip != 0:
            continue

        frame = eros.get_frame()
        sample_anno = {}
        sample_anno['img_name'] = action_name + '_' + str(fi).zfill(5) + '.jpg'
        sample_anno['ts'] = pose['ts']
        sample_anno['keypoints'] = get_movenet_keypoints(pose, args.frame_height, args.frame_width,
                                                         upperbody=args.upperbody)
        sample_anno['center'] = get_center(sample_anno['keypoints'], args.frame_height, args.frame_width)
        sample_anno['torso_size'] = get_torso_length(pose, args.frame_height, args.frame_width)
        sample_anno['keypoints_prev'] = kps_old
        sample_anno['original_sample'] = action_name

        # print(sample_anno)
        frame = cv2.GaussianBlur(frame, (args.gauss_kernel, args.gauss_kernel), 0)
        if args.write_images:
            cv2.imwrite(os.path.join(output_path, sample_anno['img_name']), frame)
        if args.dev:
            keypoints = np.reshape(sample_anno['keypoints'], [-1, 3])
            h, w = frame.shape
            for i in range(len(keypoints)):
                frame = cv2.circle(frame, [int(keypoints[i, 0] * w), int(keypoints[i, 1] * h)], 1, (255, 0, 0), 2)
            frame = cv2.circle(frame, [int(sample_anno['center'][0] * w), int(sample_anno['center'][1] * h)], 1,
                               (255, 0, 0), 4)
            cv2.imshow('', frame)
            cv2.waitKey(1)
        if args.write_video:
            video_out.write(frame)
            print('writing')
        poses_movenet.append(sample_anno)

        kps_old = sample_anno['keypoints']

    if args.write_video:
        video_out.release()

    # Writing the GT
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
        output_path_images = args.data_home + "/tester/h36m_EROS/"
        output_path_anno = args.data_home + "/tester/h36m_anno/"
    else:
        output_path_images = args.data_home + "/training/h36m_EROS/"
        output_path_anno = args.data_home + "/training/h36m_anno/"
        if args.upperbody:
            output_path_anno = args.data_home + "/training/h36m_anno_upper/"

    output_path_images = os.path.abspath(output_path_images)
    output_path_anno = os.path.abspath(output_path_anno)
    output_json = output_path_anno + '/poses.json'
    ensure_location(output_path_images)
    ensure_location(output_path_anno)

    input_data_dir = args.data_home + "/yarp/"
    input_data_dir = os.path.abspath(input_data_dir)

    dir_list = os.listdir(input_data_dir)
    already_done = setup_testing_list(output_json)

    for i in tqdm(range(len(dir_list))):
        if args.dev:
            sample = 'cam2_S11_Directions'
        else:
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
            export_to_eros(dvs_dir, data_vicon_file, output_path_images, skip=args.skip_image, args=args,
                           output_json=output_json)

        if args.dev:
            exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-eros_kernel', help='', default=8, type=int)
    parser.add_argument('-frame_width', help='', default=640, type=int)
    parser.add_argument('-frame_height', help='', default=480, type=int)
    parser.add_argument('-gauss_kernel', help='', default=7, type=int)
    parser.add_argument('-skip_image', help='', default=None)
    parser.add_argument('-data_home', help='Path to dataset folder', default='/home/ggoyal/data/h36m_cropped/',
                        type=str)
    parser.add_argument("-from_scratch", type=str2bool, nargs='?', const=True, default=False,
                        help="Write annotation file from scratch.")
    parser.add_argument("-write_annotation", type=str2bool, nargs='?', const=True, default=False,
                        help="Write annotation file.")
    parser.add_argument("-write_images", type=str2bool, nargs='?', const=True, default=False, help="Save images.")
    parser.add_argument("-write_video", type=str2bool, nargs='?', const=True, default=False, help="Save video.")
    parser.add_argument('-fps', help='', default=50, type=int)
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=False, help="Run in dev mode.")
    parser.add_argument("-ts_scaler", help='', default=12.50, type=float)
    parser.add_argument("-", type=str2bool, nargs='?', const=True, default=False,
                        help="Create annotation for upperbody only.")

    args = parser.parse_args()
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print('Catching an argumentError')
    main(args)


