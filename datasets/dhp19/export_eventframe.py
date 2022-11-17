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
import numpy as np

from datasets.utils.parsing import import_yarp_skeleton_data, batchIterator
from datasets.utils.events_representation import eventFrame
from datasets.utils.export import str2bool, checkframecount, ensure_location, get_movenet_keypoints
from bimvee.importIitYarp import importIitYarp as import_dvs


def export_to_eventFrame(data_dvs_file, data_vicon_file, output_path, skip=None, args=None):
    if skip == None:
        skip = 1
    else:
        skip = int(skip) + 1
    action_name = data_dvs_file.split(os.sep)[-2]
    cam = data_dvs_file.split(os.sep)[-1][2]

    ground_truth = import_yarp_skeleton_data(pathlib.Path(data_vicon_file))
    data_dvs = import_dvs(filePathOrName=data_dvs_file)
    data_dvs['data']['left']['dvs']['ts'] /= args.ts_scaler
    iterator = batchIterator(data_dvs['data']['left']['dvs'], ground_truth, args.n, args.offset)

    event_frame = eventFrame(frame_height=args.frame_height, frame_width=args.frame_width, n=args.n)

    if args.write_video:
        output_path_video = os.path.join(output_path,'cam'+str(cam)+'_'+action_name+'.mp4')
        print(output_path_video)
        video_out = cv2.VideoWriter(output_path_video, cv2.VideoWriter_fourcc(*'avc1'), args.fps,
                                    (args.frame_width, args.frame_height))

    for fi, (batch, skeleton, batch_size) in enumerate(iterator):
        event_frame.reset_frame()
        for ei in range(batch_size):
            event_frame.update(vx=int(batch['x'][ei]), vy=int(batch['y'][ei]))

        if fi % skip == 0:
            frame = event_frame.get_normed()
            if args.write_video:
                video_out.write(frame)
        if args.dev and fi % args.fps == 0:
            print('frame: ', fi, 'timestamp: ', skeleton['ts'])
        sample_name = 'cam' + str(cam) + '_' + action_name + '_' + str(fi) + '.jpg'
        if args.dev:
            pose = get_movenet_keypoints(skeleton, args.frame_height, args.frame_width)
            keypoints = np.reshape(pose, [-1, 3])
            h, w = frame.shape
            for i in range(len(keypoints)):
                frame = cv2.circle(frame, [int(keypoints[i, 0] * w), int(keypoints[i, 1] * h)], 1, (255, 0, 0), 2)
            cv2.imshow('', frame)
            cv2.waitKey(1)
        if args.write_images:
            cv2.imwrite(os.path.join(output_path, sample_name), frame)

    if args.write_video:
        video_out.release()

    return


def setup_testing_list(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.readlines()
    return lines

def main(args):
    input_data_dir = os.path.abspath(args.data_home+'/yarp')

    if args.dev:
        dir_list = ['S13_1_1']
    else:
        dir_list = os.listdir(input_data_dir)
        dir_list.sort()

    if args.dev:
        output_path_images = args.data_home + "/tester/eF/"
        output_path_anno = args.data_home + "/tester/anno/"
    else:
        output_path_images = args.data_home + "/training/eF/"
        output_path_anno = args.data_home + "/training/anno/"

    output_path_images = os.path.abspath(output_path_images)
    output_path_anno = os.path.abspath(output_path_anno)
    output_json = output_path_anno + '/poses.json'
    ensure_location(output_path_images)
    ensure_location(output_path_anno)

    input_data_dir = os.path.abspath(input_data_dir)

    log_file = output_path_images+'/log.txt'
    already_done = setup_testing_list(log_file)

    for i in tqdm(range(len(dir_list))):
        for cam in range(4):
            sample = dir_list[i]
            dvs_dir = os.path.join(input_data_dir, sample, f'ch{cam}dvs')
            data_vicon_file = os.path.join(input_data_dir, sample, f'ch{cam}skeleton/data.log')

            process = True
            print(str(i), "=====", sample, "=====")

            if sample in already_done:
                print('skipping: ', sample, '(already processed)')
                process = False
            elif not os.path.exists(dvs_dir):
                print('skipping: ',  dvs_dir, 'does not exist')
                process = False
            elif not os.path.exists(data_vicon_file):
                print('skipping: ',  data_vicon_file, 'does not exist')
                process = False

            if process:
                print('processing: ', sample)
                export_to_eventFrame(dvs_dir, data_vicon_file, output_path_images, skip=args.skip_image, args=args)
                with open(output_path_images+'log.txt','a') as f:
                    f.write(sample+'\n')

            print('')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='Number of events', default=7500, type=float)
    parser.add_argument('-offset', help='Number of events before skeleton', default=3250, type=float)

    parser.add_argument('-frame_width', help='', default=640, type=int)
    parser.add_argument('-frame_height', help='', default=480, type=int)
    parser.add_argument('-gauss_kernel', help='', default=7, type=int)
    parser.add_argument('-skip_image', help='', default=None)
    parser.add_argument('-data_home', help='Path to dataset folder', default='/home/ggoyal/data/DHP19/', type=str)
    parser.add_argument("-write_images", type=str2bool, nargs='?', const=True, default=True, help="Save images.")
    parser.add_argument("-write_video", type=str2bool, nargs='?', const=False, default=False, help="Save video.")
    parser.add_argument('-fps', help='', default=50, type=int)
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=False, help="Run in dev mode.")
    parser.add_argument("-ts_scaler", help='', default=1, type=float)

    args = parser.parse_args()
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print('Catching an argumentError')
    main(args)

