#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author: Franco Di Pietro and Gaurvi Goyal

LICENSE GOES HERE
"""
# Reading the GT and videos form h36 and generating Yarp compatible formats

import cdflib
import cv2
import numpy as np
import os

from datasets.utils.constants import HPECoreSkeleton
from datasets.utils.export import skeleton_to_yarp_row
from utils import parsing
from os.path import join, isfile
from tqdm import tqdm


dataset_path = '/home/ggoyal/data/h36m/extracted'
data_output_path = '/home/ggoyal/data/h36m/yarp/'
output_width = 346
output_height = 260
# subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
subs = ['S1']
all_cameras = {1: '54138969', 2: '55011271', 3: '58860488', 4: '60457274'}
cam = 2  # And maybe later camera 4.
errorlog = '/home/ggoyal/data/h36m/errorlog.txt'
OUTPUT_FRAMES = False


def write_video_and_pose(video_path, gt_path, directory_frames, directory_skl, write_frames=True, write_pose=True,
                         overwrite=False):
    # Convert the video and annotations to yarp formats.
    counter = 0
    frame_lines = []
    pose_lines = []
    vid = cv2.VideoCapture(video_path)
    cdf_file = cdflib.CDF(gt_path)
    data = (cdf_file.varget("Pose")).squeeze()
    dim = (output_width, output_height)

    if not overwrite:
        if isfile(join(dir_pose, 'data.log')):
            write_pose = False
    if write_frames:
        if not os.path.exists(directory_frames):
            os.makedirs(directory_frames)
    if not os.path.exists(directory_skl):
        os.makedirs(directory_skl)

    try:
        assert vid.isOpened()
    except AssertionError:
        return 1
    while vid.isOpened():
        frame_exists, frame = vid.read()
        if frame_exists:
            timestamp = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000  # convert timestamp to seconds
        else:
            break

        if counter > 10 and timestamp == 0.0:
            break

        try:
            skeleton = data[counter, :]
        except IndexError:
            break

        filename = 'frame_' + str(counter).zfill(10) + '.png'
        filename_full = os.path.join(directory_frames, filename)
        if write_frames:
            frame_resized = cv2.resize(src=frame, dsize=dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(filename_full, frame_resized)  # create the images

        # convert h3.6m joints order to hpecore one
        skeleton = skeleton.reshape(-1, 2)
        skeleton = parsing.h36m_to_hpecore_skeleton(skeleton)

        # rescale skeleton
        skeleton[:, 0] = skeleton[:, 0] * output_width / 1000
        skeleton[:, 1] = skeleton[:, 1] * output_height / 1000
        skeleton = np.rint(skeleton).astype(int)

        torso_size = HPECoreSkeleton.compute_torso_sizes(skeleton)

        # append strings that will be written to yarp's data.log
        frame_lines.append(" %d %.6f %s [rgb]" % (counter, timestamp, filename))
        pose_lines.append(skeleton_to_yarp_row(counter, timestamp, skeleton, torso_size=torso_size))
        counter += 1

    # info.log
    frame_linesInfo = ['Type: Image;', '[0.0] /file/ch0frames:o [connected]']
    pose_linesInfo = ['Type: Bottle;', '[0.0] /file/ch0GT50Hzskeleton:o [connected]']

    vid.release()

    if write_frames:
        parsing.writer(directory_frames, frame_lines, frame_linesInfo)
    if write_pose:
        parsing.writer(directory_skl, pose_lines, pose_linesInfo)
    # print(pose_lines)
    # print(pose_linesInfo)
    return 0


all_files = []

for sub in subs:
    files = os.listdir(join(dataset_path, sub, 'Videos'))
    for file in files:
        if all_cameras[cam] in file:
            all_files.append("%s^%s" % (sub, file))

for i in tqdm(range(len(all_files))):
    sub, file = all_files[i].split('^')
    video_file = (join(dataset_path, sub, 'Videos', file))
    pose_file = (join(dataset_path, sub, "Poses_D2_Positions", file.replace('mp4', 'cdf')))
    output_folder = ("%s_%s" % (sub, file.split('.')[0].replace(' ', '_')))
    dir_frames = join(data_output_path, output_folder, f'ch{cam}frames')
    dir_pose = join(data_output_path, output_folder, f'ch{cam}GT50Hzskeleton')
    # if isfile(join(dir_pose,'data.log')):
    #     continue
    # print((isfile(video_file),isfile(pose_file),dir_frames,dir_pose))
    if not isfile(pose_file):
        continue
    exitcode = write_video_and_pose(video_file, pose_file, dir_frames, dir_pose, write_frames=OUTPUT_FRAMES,
                                write_pose=True,overwrite=True)

    if exitcode:
        with open(errorlog, 'a') as f:
            f.write("%s" % all_files[i])


# sub, file = all_files[10].split('^')
# video_file = (join(dataset_path, sub, 'Videos', file))
# pose_file = (join(dataset_path, sub, "Poses_D2_Positions", file.replace('mp4', 'cdf')))
# output_folder = ("%s_%s" % (sub, file.split('.')[0].replace(' ', '_')))
# dir_frames = join(data_output_path, output_folder, 'ch0frames')
# dir_pose = join(data_output_path, output_folder, 'ch0GT50Hzskeleton')
# exitcode = write_video_and_pose(video_file, pose_file, dir_frames, dir_pose, write_frames=OUTPUT_FRAMES,
#                                 write_pose=True,overwrite=True)

# limbs = {0: 'PelvisC', 1: 'PelvisR', 2: 'KneeR', 3: 'AnkleR', 4: 'ToeR', 5: 'ToeROther', 6: 'PelvisL', 7: 'KneeL',
# 8: 'AnkleR', 9: 'ToeR', 10: 'ToeROther', 11: 'Spine', 12: 'SpineM', 13: 'Neck', 14: 'Head', 15: 'HeadOther',
# 16: 'NextAgain', 17: 'ShoulderL', 18: 'ElbowL', 19: 'WristL', 20: 'WristLAgain', 21: 'ThumbL', 22: 'BOHR', 23: 'BOHRAgain',
# 24: 'NeckAgainAgain', 25: 'ShoulderR', 26: 'ElbowR', 27: 'WristR', 28: 'WristAgain', 29: 'ThumbR', 30: 'BOHL', 31: 'BOHLAgain'}

# 0 = 11
# 13 = 16 = 24
# 19 = 20
# 22 = 23
# 26 = 27
# 30 = 31