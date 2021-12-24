#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author: Franco Di Pietro and Gaurvi Goyal

LICENSE GOES HERE
"""
# Reading the GT and videos form h36 and generating Yarp compatible formats

# %% Preliminaries
import numpy as np
import os
import cdflib
import cv2
from utils import parsing
from os.path import join, isfile
from tqdm import tqdm

dataset_path = '/home/ggoyal/data/h36m/extracted'
data_output_path = '/home/ggoyal/data/h36m/yarp/'
output_width = 346
output_height = 260
subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
all_cameras = {1: '54138969', 2: '55011271', 3: '58860488', 4: '60457274'}
cam = 2  # And maybe later camera 4.

def write_video_and_pose(video_path, gt_path, directory_frames, directory_skl):
    # Convert the video and annotations to yarp formats.
    counter = 0
    frame_lines = []
    pose_lines = []
    vid = cv2.VideoCapture(video_path)
    cdf_file = cdflib.CDF(gt_path)
    data = (cdf_file.varget("Pose")).squeeze()
    dim = (output_width, output_height)

    if not os.path.exists(directory_frames):
        os.makedirs(directory_frames)
    if not os.path.exists(directory_skl):
        os.makedirs(directory_skl)

    assert vid.isOpened()
    while vid.isOpened():
        frame_exists, frame = vid.read()
        if frame_exists:
            timestamp = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000  # convert timestamp to seconds
        else:
            break
        if counter > 10 and timestamp == 0.0:
            break
        frame_resized = cv2.resize(src=frame, dsize=dim, interpolation=cv2.INTER_AREA)
        filename = 'frame_' + str(counter).zfill(10) + '.png'
        filename_full = os.path.join(directory_frames, filename)
        cv2.imwrite(filename_full, frame_resized)  # create the images
        pose = data[counter, :]
        pose = pose.reshape(-1, 2)
        pose_small = parsing.h36m_to_dhp19(pose)

        pose_small[:, 0] = pose_small[:, 0] * output_width / 1000
        pose_small[:, 1] = pose_small[:, 1] * output_height / 1000
        pose_small = np.rint(pose_small)

        # # data.log
        frame_lines.append(" %d %.6f %s [rgb]" % (counter, timestamp, filename))
        pose_lines.append("%d %.6f SKLT (%s)" % (counter, timestamp, str(pose_small.reshape(-1))[1:-1]))

        counter += 1

    # info.log
    frame_linesInfo = ['Type: Image;', '[0.0] /file/ch0frames:o [connected]']
    pose_linesInfo = ['Type: Bottle;', '[0.0] /file/ch0GT50Hzskeleton:o [connected]']

    vid.release()

    print()
    parsing.writer(directory_frames, frame_lines, frame_linesInfo)
    parsing.writer(directory_skl, pose_lines, pose_linesInfo)


all_files = []

for sub in subs:
    files = os.listdir(join(dataset_path, sub, 'Videos'))
    for file in files:
        if all_cameras[cam] in file:
            all_files.append("%s^%s" % (sub, file))

print(all_files)
for i in tqdm(range(len(all_files))):
    sub, file = all_files[i].split('^')
    video_file = (join(dataset_path, sub, 'Videos', file))
    pose_file = (join(dataset_path, sub, "Poses_D2_Positions", file.replace('mp4', 'cdf')))
    output_folder = ("%s_%s" % (sub, file.split('.')[0].replace(' ', '_')))
    dir_frames = join(data_output_path, output_folder, 'ch0frames')
    dir_pose = join(data_output_path, output_folder, 'ch0GT50Hzskeleton/')
    # print((isfile(video_file),isfile(pose_file),dir_frames,dir_pose))
    if not isfile(pose_file):
        continue
    write_video_and_pose(video_file, pose_file, dir_frames, dir_pose)
    break




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
