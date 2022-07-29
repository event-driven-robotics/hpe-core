"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author:
    Gaurvi Goyal

LICENSE GOES HERE
"""

from tqdm import tqdm
import argparse
import cv2
import pathlib
import os
import re
import numpy as np


from datasets.utils.parsing import import_yarp_skeleton_data, YarpHPEIterator, batchIterator
from datasets.h36m.utils.parsing import H36mIterator, hpecore_to_movenet
from datasets.utils.events_representation import PIM
from bimvee.importIitYarp import importIitYarp as import_dvs


def checkframecount(video_file_name, gt_file_name):

    vid = cv2.VideoCapture(video_file_name)
    vid_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.release()

    num_lines = sum(1 for line in open(gt_file_name))

    if vid_frames == 0:
        print("no video frames")
        return False
    if num_lines == 0:
        print("no skeleton files")
        return False
    if vid_frames != num_lines:
        print("not correctly processed", vid_frames, num_lines)
        return False
    return True
    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def export_to_pim(data_dvs_file, data_vicon_file, video_output_path, skip=None, args=None):
    if skip == None:
        skip = 1
    else:
        skip = int(skip) + 1

    ground_truth = import_yarp_skeleton_data(pathlib.Path(data_vicon_file))
    data_dvs = import_dvs(filePathOrName=data_dvs_file[:-9])
    data_dvs['data']['left']['dvs']['ts'] /= args.ts_scaler
    iterator = batchIterator(data_dvs['data']['left']['dvs'], ground_truth)

    pim = PIM(frame_height=args.frame_height, frame_width=args.frame_width, tau=args.tau)

    video_out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'avc1'), args.fps, (args.frame_width, args.frame_height))

    for fi, (batch, skeleton, batch_size) in enumerate(iterator):
        
        for ei in range(batch_size):
             pim.update(vx=int(batch['x'][ei]), vy=int(batch['y'][ei]), p=int(batch['pol'][ei]))

        if fi % skip == 0:
            pim.perform_decay(batch['ts'][-1])
            frame = pim.get_normed_rgb()       
            video_out.write(frame)
        if fi % args.fps == 0:
            cv2.imshow('', frame)
            cv2.waitKey(1)
            if args.dev:
                print('frame: ', fi, 'timestamp: ', batch['ts'][-1])

    video_out.release()

    return

def main(args):

    input_data_dir = os.path.abspath(args.data_home)

    if args.dev:
        dir_list = ['cam2_S9_Directions']
    else:
        dir_list = os.listdir(input_data_dir)
        dir_list.sort()

    for i in tqdm(range(len(dir_list))):
        sample = dir_list[i]

        #cam = sample[3]
        #data_vicon_file = os.path.join(input_data_dir, sample, f'ch{cam}GT50Hzskeleton/data.log')

        dvs_data = os.path.join(input_data_dir, sample, 'ch0dvs/data.log')
        data_vicon_file = os.path.join(input_data_dir, sample, 'ch2GT50Hzskeleton/data.log')
        output_video_path = os.path.join(input_data_dir, sample, 'pim.mp4')

        process = True
        print("=====", sample, "=====")
        #check that the file already exists, and that it is the right size
        if os.path.exists(output_video_path):
            if checkframecount(output_video_path, data_vicon_file):
                print('skipping: ', output_video_path, '(already exists)')
                process = False
        elif not os.path.exists(dvs_data):
            print('skipping: ',  dvs_data, 'does not exist')
            process = False
        elif not os.path.exists(data_vicon_file):
            print('skipping: ',  data_vicon_file, 'does not exist')
            process = False
        
        if process:
            print('processing: ', output_video_path)
            poses_sample = export_to_pim(dvs_data, data_vicon_file, output_video_path, skip=args.skip_image, args=args)
        print('')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-tau', help='', default=1.0, type=float)
    parser.add_argument('-frame_width', help='', default=640, type=int)
    parser.add_argument('-frame_height', help='', default=480, type=int)
    parser.add_argument('-fps', help='', default=50, type=int)
    parser.add_argument('-skip_image', help='', default=None)
    parser.add_argument('-data_home', help='Path to dataset folder', default='/home/ggoyal/data/h36m/', type=str)
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=False, help="Run in dev mode.")
    parser.add_argument("-ts_scaler", help='', default=1.0, type=float)

    args = parser.parse_args()
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print('Catching an argumentError')
    main(args)

