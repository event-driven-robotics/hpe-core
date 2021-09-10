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
from dhp19.utils import mat_files

# Load env variables set on .bashrc
bimvee_path = os.environ.get('BIMVEE_PATH')
mustard_path = os.environ.get('MUSTARD_PATH')

# Add local paths
sys.path.insert(0, bimvee_path)
sys.path.insert(0, mustard_path)

# Directory with DVS (after Matlab processing) and Vicon Data 
datadir = '/home/fdipietro/hpe-data'

# Selected recording
subj, sess, mov = 13, 1, 8
recording = 'S{}_{}_{}'.format(subj, sess, mov)
cam = str(3)

# read data
poses_gt = np.load('/home/fdipietro/hpe-data/2d_Nicolo/' + recording +'/2d_poses_cam_3_7500_events.npy')
poses_pred_files = sorted(Path('/home/fdipietro/hpe-data/open-pose/' + recording).glob('*.json'))
image_files = sorted(Path('/home/fdipietro/hpe-data/grayscale/' + recording +'/' + cam +'/reconstruction').glob('*.png'))

data_events = mat_files.loadmat('/home/fdipietro/hpe-data/DVS/' + recording +'.mat')
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
    
    dataFile = directory + '/data.log'
    with open(dataFile, 'w') as f:
        for line in lines:
            f.write("%s\n" % line)
    # info.log
    linesInfo = ['Type: Image;', '[0.0] /file/ch'+ str(ch) +'frames:o [connected]']
    infoFile = directory + '/info.log'
    with open(infoFile, 'w') as f:
        for line in linesInfo:
            f.write("%s\n" % line)