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
import sys, csv
import h5py

sys.path.append('/home/ggoyal/code/hpe-core')
sys.path.append('/home/ggoyal/code/bimvee')
# sys.path.append('/home/ggoyal/code/hpe-core/example/movenet')
#
# from lib import init, MoveNet, Task
from datasets.utils.parsing import import_yarp_skeleton_data, batchIterator
from datasets.utils.events_representation import EROS, eventFrame
# from datasets.h36m.utils.parsing import movenet_to_hpecore
from datasets.utils.export import ensure_location, str2bool, get_movenet_keypoints, get_center
from bimvee.importIitYarp import importIitYarp as import_dvs
from bimvee.importIitYarp import importIitYarpBinaryDataLog

from pycore.moveenet import init, MoveNet, Task

from pycore.moveenet.config import cfg
from pycore.moveenet.visualization.visualization import add_skeleton, movenet_to_hpecore
from pycore.moveenet.utils.utils import arg_parser
from pycore.moveenet.task.task_tools import image_show, write_output, superimpose


def create_ts_list(fps, ts):
    out = dict()
    out['ts'] = list()
    x = np.arange(ts[0], ts[-1], 1 / fps)
    for i in x:
        out['ts'].append(i)
    return out

def import_h5(filename):
    hf = h5py.File(filename, 'r')
    data = np.array(hf["events"][:]) #dataset_name is same as hdf5 object name
    container = {}
    container['data'] = {}
    container['data']['ch0'] = {}
    container['data']['ch0']['dvs'] = {}
    try:
        container['data']['ch0']['dvs']['ts'] = (data[:, 0]-data[0, 0])*1e-6
    except IndexError:
        print('Error reading .h5 file.')
        exit()
    container['data']['ch0']['dvs']['x'] = data[:, 1]
    container['data']['ch0']['dvs']['y'] = data[:, 2]
    container['data']['ch0']['dvs']['pol'] = data[:, 3].astype(bool)
    return container

def get_representation(rep_name, args):
    if rep_name == 'eros':
        rep = EROS(kernel_size=args.eros_kernel, frame_width=args.frame_width, frame_height=args.frame_height)
    elif rep_name == 'ef':
        rep = eventFrame(frame_height=args.frame_height, frame_width=args.frame_width, n=args.n)
    else:
        print('Representation not found for this setup.')
        exit()
    return rep

def import_file(data_dvs_file):
    filename = os.path.basename(data_dvs_file)
    if filename == 'binaryevents.log':
        data_dvs = importIitYarpBinaryDataLog(filePathOrName=data_dvs_file)
    elif os.path.splitext(filename)[1] == '.h5':
        data_dvs = import_h5(data_dvs_file)
    else:
        data_dvs = import_dvs(filePathOrName=os.path.join(data_dvs_file))
    print('File imported.')
    return data_dvs

def process(data_dvs_file, output_path, skip=None, args=None):
    init(cfg)
    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    run_task = Task(cfg, model)
    run_task.modelLoad(cfg['ckpt'])

    if skip == None:
        skip = 1
    else:
        skip = int(skip) + 1

    data_dvs = import_file(data_dvs_file)
    channel = list(data_dvs['data'].keys())[0]
    data_dvs['data'][channel]['dvs']['ts'] /= args.ts_scaler
    data_ts = create_ts_list(args.fps, data_dvs['data'][channel]['dvs']['ts'])


    iterator = batchIterator(data_dvs['data'][channel]['dvs'], data_ts)
    rep = get_representation(args.rep, args)

    if output_path != None:
        video_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), args.fps,
                                    (args.frame_width, args.frame_height))

    for fi, (events, pose, batch_size) in enumerate(iterator):
        rep.reset_frame()
        if fi % 100 == 0:
            print('frame: ', fi, '/', len(data_ts['ts']))

        if args.stop:
            if fi > args.stop:
                break

        for ei in range(batch_size):
            rep.update(vx=int(events['x'][ei]), vy=int(events['y'][ei]))
        if fi % skip != 0:
            continue

        frame = rep.get_frame()

        if args.rep == 'eros':
            frame = cv2.GaussianBlur(frame, (args.gauss_kernel, args.gauss_kernel), 0)

        pre = run_task.predict_online(frame, ts=data_ts['ts'][fi])
        output = np.concatenate((pre['joints'].reshape([-1,2]), pre['confidence'].reshape([-1,1])), axis=1)
        if args.visualise or output_path != None:
            frame = add_skeleton(frame, output, (0, 0, 255), True, normalised=False)
            if args.visualise:
                cv2.imshow('', frame)
                cv2.waitKey(1)
            if output_path != None:
                video_out.write(frame)

    if output_path != None:
        video_out.release()

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-eros_kernel', help='EROS kernel size', default=8, type=int)
    parser.add_argument('-frame_width', help='', default=640, type=int)
    parser.add_argument('-frame_height', help='', default=480, type=int)
    parser.add_argument('-gauss_kernel', help='Gaussian filter for EROS', default=7, type=int)
    parser.add_argument('-input', help='Path to input folder (with the data.log file in it)', default=None, type=str)
    parser.add_argument("-write_video", type=str, default=None, help="Set path with file name to save video")
    parser.add_argument("-ckpt", type=str, default='models/e97_valacc0.81209.pth', help="path to the ckpt. Default: MoveEnet checkpoint.")
    parser.add_argument('-fps', help='Output frame rate', default=50, type=int)
    parser.add_argument('-stop', help='Set to an integer value to stop early after these frames', default=None, type=int)
    parser.add_argument('-rep', help='Representation eros or ef', default='eros', type=str)
    parser.add_argument('-n', help='Number of events in constant count event frame [7500]', default=7500, type=int)
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=True, help="Run in dev mode.")
    parser.add_argument("-ts_scaler", help='', default=1.0, type=float)
    parser.add_argument('-visualise', type=str2bool, nargs='?', default=True, help="Visualise Results [TRUE]")


    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print('Catching an argumentError')
        exit()

    cfg['ckpt'] = args.ckpt
    if args.dev:
        args.input = '/home/ggoyal/data/h36m_cropped/tester/h5/cam2_S1_Directions_1/Directions.h5'
        args.write_video = '/home/ggoyal/data/tester.mp4'
        args.stop = 100

    if args.input == None:
        print('Please set input path')
        exit()

    if args.write_video is not None:
        output_path = os.path.abspath(args.write_video)
        ensure_location(os.path.dirname(output_path))
    else:
        output_path = None
    input_data_dir = os.path.abspath(args.input)

    print("=====", input_data_dir, "=====")
    if not os.path.exists(input_data_dir):
        print(input_data_dir, 'does not exist')
    else:
        process(input_data_dir, output_path, args=args)


