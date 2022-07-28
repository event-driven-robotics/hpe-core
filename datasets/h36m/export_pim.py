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
from datasets.utils.events_representation import PIM
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


def export_to_pim(data_dvs_file, data_vicon_file, output_path_images, skip=None, args=None):
    if skip == None:
        skip = 1
    else:
        skip = int(skip) + 1

    data_vicon = importSkeletonData(data_vicon_file)
    data_dvs = import_dvs(filePathOrName=data_dvs_file)

    iterator = H36mIterator(data_dvs['data']['left']['dvs'], data_vicon,time_factor=12.5)
    pim = PIM(frame_height=args.frame_height, frame_width=args.frame_width, tau=args.tau)

    vid = cv2.VideoCapture(output_path_images)
    sample_name = data_dvs_file.split(os.sep)[-2]
    video_out = cv2.VideoWriter(f'{output_path_images}/pim.mp4', cv2.VideoWriter_fourcc(*'h264'), args.fps, (args.frame_width, args.frame_height))
    poses_movenet = []

    for fi, (events, pose, ts) in enumerate(iterator):
        if args.dev:
            print('frame: ', fi, 'timestamp: ', ts)
        for ei in range(len(events)):
            pim.update(vx=int(events[ei, 0]), vy=int(events[ei, 1]), p=events[ei, 2])

        if fi % skip == 0:
            pim.perform_decay(ts)
            frame = pim.get_normed_rgb()       
            video_out.write(frame)
            cv2.imshow('', frame)
            cv2.waitKey(1)

    video_out.release()
            # kps_old = sample_anno['keypoints']
    return

def main(args):

    input_data_dir = args.data_home
    input_data_dir = os.path.abspath(input_data_dir)

    dir_list = os.listdir(input_data_dir)

    dir_list.sort()

    # print(already_done)
    for i in tqdm(range(len(dir_list))):

        #if it begins with _ ignore for now

        if args.dev:
            sample = 'cam2_S9_Directions'
        else:
            sample = dir_list[i]

        #cam = sample[3]
        dvs_dir = os.path.join(input_data_dir, sample, 'ch0dvs')
        #data_vicon_file = os.path.join(input_data_dir, sample, f'ch{cam}GT50Hzskeleton/data.log')
        data_vicon_file = os.path.join(input_data_dir, sample, f'ch2GT50Hzskeleton/data.log')

        print(str(i), sample)
        print(data_vicon_file)

        output_video_path = os.path.join(input_data_dir, sample)
        print(output_video_path)

        if os.path.exists(dvs_dir) and os.path.exists(data_vicon_file):
            print('Files found.')
            poses_sample = export_to_pim(dvs_dir, data_vicon_file, output_video_path, skip=args.skip_image, args=args)


        else:
            print(f"File {sample} not found.")
            #if args.dev:
            print(os.path.exists(dvs_dir))
            print(os.path.exists(data_vicon_file))
            print(' ')

        if args.dev:
            exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-tau', help='', default=1.0, type=float)
    parser.add_argument('-frame_width', help='', default=640, type=int)
    parser.add_argument('-frame_height', help='', default=480, type=int)
    parser.add_argument('-fps', help='', default=50, type=int)
    parser.add_argument('-skip_image', help='', default=None)
    parser.add_argument('-data_home', help='Path to dataset folder', default='/home/ggoyal/data/h36m/', type=str)
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=True, help="Run in dev mode.")

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

