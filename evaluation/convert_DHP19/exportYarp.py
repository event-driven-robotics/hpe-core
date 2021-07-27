#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author: Franco Di Pietro 

LICENSE GOES HERE
"""

# %% Preliminaries
import numpy as np
import os
from os.path import join
import sys
import scipy.io as spio

# Load env variables set on .bashrc
bimvee_path = os.environ.get('BIMVEE_PATH')
mustard_path = os.environ.get('MUSTARD_PATH')

# Add local paths
sys.path.insert(0, bimvee_path)
sys.path.insert(0, mustard_path)

# Directory with DVS (after Matlab processing) and Vicon Data 
datadir = '/home/fdipietro/hpe-data'

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

# Export to yarp (only DVS)
from bimvee import exportIitYarp

numSubjects = 17;
numSessions = 5;
pathRawData = '/home/fdipietro/hpe-data/DVS/'
exportPath = '/home/fdipietro/hpe-data/yarp/'

for subj in range(1,numSubjects+1):
    for sess in range(1,numSessions+1):
        
        if(sess == 1):
            numMovements = 8;
        elif(sess == 2):
            numMovements = 6;
        elif(sess == 3):
            numMovements = 6;
        elif(sess == 4):
            numMovements = 6;
        elif(sess == 5):
            numMovements = 7;
            
        for mov in range(1,numMovements+1):
            # Selected recording
            
            datafile = 'S{}_{}_{}'.format(subj, sess, mov)+'.mat'
            if os.path.exists(pathRawData+datafile):
                print('Exporting file ' + pathRawData+datafile + ' ...')
                # datafile += '.mat'
                
                outfile = 'S{}_{}_{}'.format(subj, sess, mov)
                
                
                # Load DVS data
                DVS_dir = join(datadir, 'DVS/')
                dataDvs = loadmat(join(DVS_dir,datafile))
                
                # Build container
                info = {}
                # info['filePathOrName'] = ''
                container = {}
                container['info'] = info
                container['data'] = {}
                startTime = dataDvs['out']['extra']['startTime']
                for i in range(4):
                    container['data']['ch'+str(i)] = dataDvs['out']['data']['cam'+str(i)]
                    container['data']['ch'+str(i)]['dvs']['x'] = container['data']['ch'+str(i)]['dvs']['x']  - 1  - 346*i
                    container['data']['ch'+str(i)]['dvs']['y'] = container['data']['ch'+str(i)]['dvs']['y']  - 1
                    container['data']['ch'+str(i)]['dvs']['ts'] = (container['data']['ch'+str(i)]['dvs']['ts'] - startTime) * 1e-6
                    container['data']['ch'+str(i)]['dvs']['pol'] = np.array(container['data']['ch'+str(i)]['dvs']['pol'], dtype=bool)
        
                exportIitYarp.exportIitYarp(container, exportFilePath= exportPath+outfile, protectedWrite = True)
            else:
                print('File ' + pathRawData+datafile + ' does not exists')