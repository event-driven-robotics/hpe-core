#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Event-driven Perception for Robotics
Authors:
    Franco Di Pietro
    Nicolo' Carissimi

LICENSE GOES HERE
"""

# %% Preliminaries
import numpy as np
import os
import sys

from os.path import join

import datasets.dhp19.utils.mat_files as mat_utils

# Load env variables set on .bashrc
bimvee_path = os.environ.get('BIMVEE_PATH')
mustard_path = os.environ.get('MUSTARD_PATH')

# Add local paths
sys.path.insert(0, bimvee_path)
sys.path.insert(0, mustard_path)

# Directory with DVS (after Matlab processing) and Vicon Data 
datadir = '/home/fdipietro/hpe-data'

# Export to yarp (only DVS)
from bimvee import exportIitYarp

numSubjects = 17
numSessions = 5
pathRawData = '/home/fdipietro/hpe-data/DVS/'
exportPath = '/home/fdipietro/hpe-data/yarp/'

for subj in range(1, numSubjects + 1):
    for sess in range(1, numSessions + 1):

        if sess == 1:
            numMovements = 8
        elif sess == 2:
            numMovements = 6
        elif sess == 3:
            numMovements = 6
        elif sess == 4:
            numMovements = 6
        elif sess == 5:
            numMovements = 7

        for mov in range(1, numMovements + 1):
            # Selected recording

            datafile = 'S{}_{}_{}'.format(subj, sess, mov) + '.mat'
            if os.path.exists(pathRawData + datafile):
                print('Exporting file ' + pathRawData + datafile + ' ...')
                # datafile += '.mat'

                outfile = 'S{}_{}_{}'.format(subj, sess, mov)

                # Load DVS data
                DVS_dir = join(datadir, 'DVS/')
                dataDvs = mat_utils.loadmat(join(DVS_dir, datafile))

                # Build container
                info = {}
                # info['filePathOrName'] = ''
                container = {}
                container['info'] = info
                container['data'] = {}
                startTime = dataDvs['out']['extra']['startTime']
                for i in range(4):
                    container['data']['ch' + str(i)] = dataDvs['out']['data']['cam' + str(i)]
                    container['data']['ch' + str(i)]['dvs']['x'] = container['data']['ch' + str(i)]['dvs'][
                                                                       'x'] - 1 - 346 * i
                    container['data']['ch' + str(i)]['dvs']['y'] = container['data']['ch' + str(i)]['dvs']['y'] - 1
                    container['data']['ch' + str(i)]['dvs']['ts'] = (container['data']['ch' + str(i)]['dvs'][
                                                                         'ts'] - startTime) * 1e-6
                    container['data']['ch' + str(i)]['dvs']['pol'] = np.array(
                        container['data']['ch' + str(i)]['dvs']['pol'], dtype=bool)

                exportIitYarp.exportIitYarp(container, exportFilePath=exportPath + outfile, protectedWrite=True)
            else:
                print('File ' + pathRawData + datafile + ' does not exists')
