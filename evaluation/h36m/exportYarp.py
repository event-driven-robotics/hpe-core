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
import sys
import cdflib
import cv2
from os.path import join

video_path = '/home/icub/data/h36m/extracted/S1/Videos/Directions.55011271.mp4'
gt_path = '/home/icub/data/h36m/extracted/S1/Poses_D2_Positions/Directions.55011271.cdf'
base_output_folder = '/home/icub/data/h36m/testers/'
output_width='346'
output_height='260'
dataset_path = '/home/icub/data/h36m/'
subs = ['S1','S5','S6','S7','S8','S9','S11']
all_cameras = {1:'54138969',2:'55011271',3:'58860488',4:'60457274'}
cam = 2 # And maybe later camera 4.

# all_data_paths = [os.path.join(dataset_path,sub,'Videos') for sub in subs]
# print(all_data_paths)
# print(all_cameras)
#
# files = list([])
# output_folders = list([])
# for sub in subs:
#     path = os.path.join(dataset_path,sub,'Videos')
#     temp_name = path + '/*'+all_cameras[cam]+'.mp4'
#     files_temp = glob.glob(temp_name)
#     [files.append(file.replace(' ','\ ')) for file in files_temp]
#     [output_folders.append(sub+'_'+os.path.basename(file).split('.')[0].replace(' ','_')) for file in files_temp]
#
# # Check that everything went well
# assert len(files) == len(output_folders)

dim = (output_width, output_height)
# directory = os.path.join(dataset_path, 'yarp/S1_Purchases/ch0frames/')
# if not os.path.exists(directory):
#     os.makedirs(directory)
counter = 0
lines = []
frame_num = 0
vid = cv2.VideoCapture(video_path)

assert vid.isOpened()
while vid.isOpened():
    frame_exists, frame = vid.read()
    if frame_exists:
        timestamp = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000  # convert timestamp to seconds
    else:
        break
    if counter > 10 and timestamp == 0.0:
        continue
    # frame_resized = cv2.resize(src=frame, dsize=dim, interpolation=cv2.INTER_AREA)
    # filename = 'frame_' + str(counter).zfill(10) + '.png'
    # filename_full = os.path.join(directory, filename)
    # cv2.imwrite(filename_full, frame_resized)
    frame_num +=1
    # data.log
    # lines.append(str(counter) + ' ' + str(timestamp) + ' ' + str(filename) + '  [rgb]')

    counter += 1

vid.release()
print(frame_num)

cdf_file = cdflib.CDF(gt_path)
print(cdf_file.varinq('Pose')['Dim_Sizes'][0])