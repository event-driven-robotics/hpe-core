#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author: Franco Di Pietro 

This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with 
this program. If not, see <https://www.gnu.org/licenses/>.
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

# Selected recording
subj, sess, mov = 1, 2, 1
datafile = 'S{}_{}_{}'.format(subj, sess, mov)+'.mat'


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

# %% Load DVS data
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

    

# %% Load Vicon data
Vicon_dir = join(datadir, 'Vicon/')
dataVicon = loadmat(join(Vicon_dir,datafile))

dt = (dataDvs['out']['extra']['ts'][-1]-startTime)/np.shape(dataVicon['XYZPOS']['head'])[0]
thz = np.arange(dataDvs['out']['extra']['ts'][0]-startTime, dataDvs['out']['extra']['ts'][-1]-startTime+dt, dt) * 1e-6 # Vicon timestams @ 100Hz


# %% Vicon 3D -> 2D
#  % Load P Matrix
P_mat_dir = join(datadir, 'P_matrices/')

# constant parameters
H = 260; W = 344; num_joints = 13

# % Load camera matrices, camera centers, input and label files, and import CNN mode
P_mat_cam1 = np.load(join(P_mat_dir,'P1.npy'))
P_mat_cam2 = np.load(join(P_mat_dir,'P2.npy'))
P_mat_cam3 = np.load(join(P_mat_dir,'P3.npy'))
P_mat_cam4 = np.load(join(P_mat_dir,'P4.npy'))
P_mats = [P_mat_cam1, P_mat_cam2, P_mat_cam3, P_mat_cam4]
cameras_pos = np.load(join(P_mat_dir,'camera_positions.npy'))

def get_all_joints(viconData, idx):
    viconMat = np.zeros([13,3],dtype=viconData['XYZPOS']['head'].dtype)   
    for i in range(0,len(viconData['XYZPOS'])):
        viconMat[i,:] = viconData['XYZPOS'][list(viconData['XYZPOS'])[i]][idx]
    return viconMat

def get_2Dcoords_and_heatmaps_label(vicon_xyz, ch_idx):
    # " From 3D label, get 2D label coordinates and heatmaps for selected camera "
    if ch_idx==1: P_mat_cam = np.load(join(P_mat_dir,'P1.npy'))
    elif ch_idx==3: P_mat_cam = np.load(join(P_mat_dir,'P2.npy'))
    elif ch_idx==2: P_mat_cam = np.load(join(P_mat_dir,'P3.npy'))
    elif ch_idx==0: P_mat_cam = np.load(join(P_mat_dir,'P4.npy'))
    # use homogeneous coordinates representation to project 3d XYZ coordinates to 2d UV pixel coordinates.
    vicon_xyz_homog = np.concatenate([vicon_xyz, np.ones([1,13])], axis=0)
    coord_pix_homog = np.matmul(P_mat_cam, vicon_xyz_homog)
    coord_pix_homog_norm = coord_pix_homog / coord_pix_homog[-1]
    u = coord_pix_homog_norm[0]
    v = H - coord_pix_homog_norm[1] # flip v coordinate to match the image direction
    # mask is used to make sure that pixel positions are in frame range.
    mask = np.ones(u.shape).astype(np.float32)
    mask[np.isnan(u)] = 0; mask[np.isnan(v)] = 0
    mask[u>W] = 0; mask[u<=0] = 0; mask[v>H] = 0; mask[v<=0] = 0
    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    return np.stack((v,u), axis=-1), mask

size = dataVicon['XYZPOS']['head'].shape[0]
dataType = dataVicon['XYZPOS']['head'].dtype
joints = {}
for ch in range(4):
    joints['ch'+str(ch)] = {}
    for key in dataVicon['XYZPOS']:
        joints['ch'+str(ch)][key] = np.empty((size,2), dtype = 'uint16')
    for i in range(len(thz)-1):
        viconMat = get_all_joints(dataVicon, i)
        y_2d, gt_mask = get_2Dcoords_and_heatmaps_label(np.transpose(viconMat), ch)
        y_2d_float = y_2d.astype(np.uint16)
        for j in range(13):
            joints['ch'+str(ch)][list(joints['ch'+str(ch)])[j]][i,1] = y_2d_float[j,0].astype(np.uint16)
            joints['ch'+str(ch)][list(joints['ch'+str(ch)])[j]][i,0] = y_2d_float[j,1].astype(np.uint16)
    joints['ch'+str(ch)]['ts'] = thz


# %% Build  DVS+GT container
for ch in range(4):
    # Change dvs data to dhp19
    container['data']['ch'+str(ch)]['dhp19'] = container['data']['ch'+str(ch)].pop('dvs')
    # Add GT data
    container['data']['ch'+str(ch)]['dhp19']['skeleton'] = {}
    container['data']['ch'+str(ch)]['dhp19']['skeleton']['gt'] = joints['ch'+str(ch)]
            
    
# %% Plot joint vs t
import matplotlib.pyplot as plt

plt.close('all')
fig, ax = plt.subplots(2)
ch = 3 # choose cahnnel to plot
for j in range(13):
    ax[0].plot(thz, joints['ch'+str(ch)][list(joints['ch'+str(ch)])[j]][:,0],)
    ax[0].set_ylabel('x coordinate [px]', fontsize=12, labelpad=5)
    ax[1].plot(thz, joints['ch'+str(ch)][list(joints['ch'+str(ch)])[j]][:,1])
    ax[1].set_ylabel('y coordinate [px]', fontsize=12, labelpad=5)
    
for i in range(2):
    ax[i].legend(list(joints['ch'+str(ch)]), loc='upper right', fontsize=11)
    ax[i].set_xlabel('time [sec]', fontsize=12, labelpad=-5)
    ax[i].grid()
fig.suptitle('Ground truth 2D position for ch'+str(ch)+' camera', fontsize=18)


# %% Generate a second skeleton with noise to see if comparisson works
import copy

noisySkt = copy.deepcopy(joints)
for ch in range(4):
    for key in noisySkt['ch'+str(ch)]:
        if key != 'ts':
            noise = np.int_(np.random.normal(0, 3,  noisySkt['ch'+str(ch)][key].shape))
            noisySkt['ch'+str(ch)][key] =  noisySkt['ch'+str(ch)][key] + noise

for ch in range(4):
    container['data']['ch'+str(ch)]['dhp19']['skeleton']['test'] = noisySkt['ch'+str(ch)]

# %% Start Mustard
cwd = os.getcwd()

import threading
import mustard
app = mustard.Mustard()
thread = threading.Thread(target=app.run)
thread.daemon = True
thread.start()


# %% Once mustard is open, undo the change of working directory
os.chdir(cwd)


#%% Visualize data
app.setData(container)


# %% Export to yarp (only DVS)
from bimvee import exportIitYarp

datafile = 'S{}_{}_{}'.format(subj, sess, mov)

exportIitYarp.exportIitYarp(container, 
              exportFilePath= '/home/fdipietro/hpe-data'+datafile, 
              pathForPlayback= '/home/fdipietro/hpe-data'+datafile)