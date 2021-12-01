#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:59:09 2021

@author: fdipietro
"""

# %% Preliminaries
import numpy as np
import os
from os.path import join
import sys
from pathlib import Path
from datasets.dhp19.utils import mat_files

# Load env variables set on .bashrc
bimvee_path = os.environ.get('BIMVEE_PATH')
mustard_path = os.environ.get('MUSTARD_PATH')

# Add local paths
sys.path.insert(0, bimvee_path)
sys.path.insert(0, mustard_path)

# Directory with DVS (after Matlab processing) and Vicon Data 
datadir = '/home/ggoyal/data/h36m/'

input_type = 'videos' # 'images' or 'videos'

def writer(directory,datalines,infolines):

    if not os.path.exists(directory):
        os.makedirs(directory)

    # datafile.log
    dataFile = directory + '/data.log'
    with open(dataFile, 'w') as f:
        for line in datalines:
            f.write("%s\n" % line)
    # info.log

    infoFile = directory + '/info.log'
    with open(infoFile, 'w') as f:
        for line in infolines:
            f.write("%s\n" % line)


if input_type == 'images':
    # Selected recording
    subj, sess, mov = 13, 1, 8
    recording = 'S{}_{}_{}'.format(subj, sess, mov)
    cam = str(3)

    # read data
    poses_gt = np.load('/home/fdipietro/hpe-data/2d_Nicolo/' + recording +'/2d_poses_cam_3_7500_events.npy')
    poses_pred_files = sorted(Path('/home/fdipietro/hpe-data/open-pose/' + recording).glob('*.json'))
    image_files = sorted(Path('/home/fdipietro/hpe-data/grayscale/' + recording +'/' + cam +'/reconstruction').glob('*.png'))

    data_events = mat_files.loadmat('/home/fdipietro/hpe-data/DVS/' + recording + '.mat')
    startTime = data_events['out']['extra']['startTime']
    t_op = np.loadtxt('/home/fdipietro/hpe-data/grayscale/' + recording +'/' + cam +'/reconstruction/timestamps.txt', dtype = np.float64)
    t_op = (t_op-startTime)*1e-6

    imgs_dir = join(datadir, 'grayscale/'+recording+'/')
    # dataDvs = loadmat(join(DVS_dir,datafile))

    from os import walk


    for ch in range(4):
        # read all image file names
        path = join(imgs_dir, str(ch)+'/reconstruction/')
        filenames = next(walk(path), (None, None, []))[2]
        filenames.remove('timestamps.txt')
        filenames = sorted(filenames)
        # read timestamps
        t_op = np.loadtxt('/home/fdipietro/hpe-data/grayscale/' + recording +'/' + str(ch) +'/reconstruction/timestamps.txt', dtype = np.float64)
        t_op = (t_op-startTime)*1e-6
        # create dirs
        directory = '/home/fdipietro/hpe-data/yarp_imgs/'+ str(recording)+ '/ch' + str(ch) + 'frames'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # datafile.log
        lines = [None] * len(t_op)
        for i in range(len(t_op)):
             lines[i] = str(i+1) + ' ' + str(t_op[i]) + ' ' + str(filenames[i]) + '  [mono]'

        # info.log
        linesInfo = ['Type: Image;', '[0.0] /file/ch'+ str(ch) +'frames:o [connected]']

        writer(directory, lines, linesInfo)

if input_type == 'videos':
    import cv2

    vid_file = '/home/ggoyal/data/h36m/extracted/S1/Videos/Purchases.55011271.mp4'
    output_width = 346
    output_height = 260
    dim = (output_width, output_height)
    directory = os.path.join(datadir,'yarp/S1_Purchases/ch0frames/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    counter = 0
    lines = []

    vid = cv2.VideoCapture(vid_file)

    assert vid.isOpened()
    while vid.isOpened():
        frame_exists, frame = vid.read()
        if frame_exists:
            timestamp = vid.get(cv2.CAP_PROP_POS_MSEC)/1000 # convert timestamp to seconds
        else:
            break
        if counter>10 and timestamp == 0.0:
            continue
        frame_resized = cv2.resize(src=frame, dsize=dim, interpolation=cv2.INTER_AREA)
        filename = 'frame_' + str(counter).zfill(10) + '.png'
        filename_full = os.path.join(directory,filename)
        cv2.imwrite(filename_full,frame_resized)

        # data.log
        lines.append(str(counter) + ' ' + str(timestamp) + ' ' + str(filename) + '  [rgb]')

        counter += 1

    vid.release()

    # info.log
    linesInfo = ['Type: Image;', '[0.0] /file/ch0frames:o [connected]']

    writer(directory, lines, linesInfo)
    print(lines)
    print(linesInfo)