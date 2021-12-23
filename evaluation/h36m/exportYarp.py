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

video_path = '/home/ggoyal/data/h36m/extracted/S1/Videos/Directions.55011271.mp4'
gt_path = '~/data/h36m/extracted/S1/Poses_D2_Positions/Directions.55011271.cdf'
base_output_folder = '/home/icub/data/h36m/testers/'
output_width='346'
output_height='260'
dataset_path = '/home/icub/data/h36m/'
subs = ['S1','S5','S6','S7','S8','S9','S11']
all_cameras = {1:'54138969',2:'55011271',3:'58860488',4:'60457274'}
cam = 2 # And maybe later camera 4.

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
    break
    # if counter > 10 and timestamp == 0.0:
    #     continue
    # frame_resized = cv2.resize(src=frame, dsize=dim, interpolation=cv2.INTER_AREA)
    # filename = 'frame_' + str(counter).zfill(10) + '.png'
    # filename_full = os.path.join(directory, filename)
    # cv2.imwrite(filename_full, frame_resized)
    frame_num +=1
    # data.log
    # lines.append(str(counter) + ' ' + str(timestamp) + ' ' + str(filename) + '  [rgb]')
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    counter += 1

vid.release()
print(frame_num)
print(counter)

cdf_file = cdflib.CDF(gt_path)
print(cdf_file.varinq('Pose')['Dim_Sizes'][0])
print(cdf_file.varinq('Pose'))
xpla = (cdf_file.varget("Pose"))
# print(x.squeeze()[0:10,:])
# print(width, height)



data = xpla[:,0,:].squeeze()
print(data)
data = data.reshape(-1,2)
print(len(data))
count = 0
limb = [31]
limb2 = [32]
for x,y in data:
    if count in limb:
        frame = cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 255, 0), thickness=10)
    if count in limb2:
        frame = cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=5)


    count +=1
cv2.imshow('frame',frame)

cv2.waitKey(0)

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