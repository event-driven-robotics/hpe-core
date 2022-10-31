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

from datasets.utils.parsing import import_yarp_skeleton_data, batchIterator
from datasets.utils.events_representation import eventFrame
from datasets.utils.export import str2bool, checkframecount
from bimvee.importIitYarp import importIitYarp as import_dvs


def export_to_eventFrame(data_dvs_file, data_vicon_file, video_output_path, skip=None, args=None):
    if skip == None:
        skip = 1
    else:
        skip = int(skip) + 1

    ground_truth = import_yarp_skeleton_data(pathlib.Path(data_vicon_file))
    data_dvs = import_dvs(filePathOrName=data_dvs_file[:-9])
    data_dvs['data']['left']['dvs']['ts'] /= args.ts_scaler
    iterator = batchIterator(data_dvs['data']['left']['dvs'], ground_truth, args.n, args.offset)

    event_frame = eventFrame(frame_height=args.frame_height, frame_width=args.frame_width, n=args.n)

    video_out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'avc1'), args.fps,
                                (args.frame_width, args.frame_height))

    for fi, (batch, skeleton, batch_size) in enumerate(iterator):
        event_frame.reset_frame()
        for ei in range(batch_size):
            event_frame.update(vx=int(batch['x'][ei]), vy=int(batch['y'][ei]))

        if fi % skip == 0:
            frame = event_frame.get_normed()
            video_out.write(frame)
        if fi % args.fps == 0:
            cv2.imshow('', frame)
            cv2.waitKey(10)
            if args.dev:
                print('frame: ', fi, 'timestamp: ', skeleton['ts'])

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

        # cam = sample[3]
        # data_vicon_file = os.path.join(input_data_dir, sample, f'ch{cam}GT50Hzskeleton/data.log')

        dvs_data = os.path.join(input_data_dir, sample, 'ch0dvs/data.log')
        data_vicon_file = os.path.join(input_data_dir, sample, 'ch2GT50Hzskeleton/data.log')
        output_video_path = os.path.join(input_data_dir, sample, 'eventFrame.mp4')

        process = True
        print("=====", sample, "=====")
        # check that the file already exists, and that it is the right size
        if os.path.exists(output_video_path):
            if checkframecount(output_video_path, data_vicon_file):
                print('skipping: ', output_video_path, '(already exists)')
                process = False
        elif not os.path.exists(dvs_data):
            print('skipping: ', dvs_data, 'does not exist')
            process = False
        elif not os.path.exists(data_vicon_file):
            print('skipping: ', data_vicon_file, 'does not exist')
            process = False

        if process:
            print('processing: ', output_video_path)
            export_to_eventFrame(dvs_data, data_vicon_file, output_video_path, skip=args.skip_image, args=args)
        print('')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='Number of events', default=7500, type=float)
    parser.add_argument('-offset', help='Number of events before skeleton', default=3250, type=float)
    parser.add_argument('-frame_width', help='', default=640, type=int)
    parser.add_argument('-frame_height', help='', default=480, type=int)
    parser.add_argument('-fps', help='', default=50, type=int)
    parser.add_argument('-skip_image', help='', default=None)
    parser.add_argument('-data_home', help='Path to dataset folder', default='/home/ggoyal/data/h36m_cropped/yarp/', type=str)
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=True, help="Run in dev mode.")
    parser.add_argument("-ts_scaler", help='', default=12.5, type=float)

    args = parser.parse_args()
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print('Catching an argumentError')
    main(args)

