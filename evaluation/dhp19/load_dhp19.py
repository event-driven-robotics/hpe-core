#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author: Franco Di Pietro 

LICENSE GOES HERE
"""

# %% Preliminaries
import math
import numpy as np
import os
import sys

from os.path import join

import datasets.utils.mat_files as mat_utils

# Load env variables set on .bashrc
bimvee_path = os.environ.get('BIMVEE_PATH')
mustard_path = os.environ.get('MUSTARD_PATH')

# Add local paths
sys.path.insert(0, bimvee_path)
sys.path.insert(0, mustard_path)

# Directory with DVS (after Matlab processing) and Vicon Data 
datadir = '/data/dhp19'

# Selected recording
subj, sess, mov = 1, 1, 1
datafile = 'S{}_{}_{}'.format(subj, sess, mov) + '.mat'

# %% Load DVS data
DVS_dir = join(datadir, 'events_preprocessed')
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
    container['data']['ch' + str(i)]['dvs']['x'] = container['data']['ch' + str(i)]['dvs']['x'] - 1 - 346 * i
    container['data']['ch' + str(i)]['dvs']['y'] = container['data']['ch' + str(i)]['dvs']['y'] - 1
    container['data']['ch' + str(i)]['dvs']['ts'] = (container['data']['ch' + str(i)]['dvs']['ts'] - startTime) * 1e-6
    container['data']['ch' + str(i)]['dvs']['pol'] = np.array(container['data']['ch' + str(i)]['dvs']['pol'],
                                                              dtype=bool)

# %% Load Vicon data
Vicon_dir = join(datadir, 'vicon')
dataVicon = mat_utils.loadmat(join(Vicon_dir, datafile))

# dt = (dataDvs['out']['extra']['ts'][-1]-startTime)/np.shape(dataVicon['XYZPOS']['head'])[0]
dt = 10000
thz = np.arange(dataDvs['out']['extra']['ts'][0] - startTime, dataDvs['out']['extra']['ts'][-1] - startTime + dt,
                dt) * 1e-6  # Vicon timestams @ 100Hz
diff = len(thz) - dataVicon['XYZPOS']['head'].shape[0]
if diff > 0:
    thz = thz[:-diff]

# %% Vicon 3D -> 2D
#  % Load P Matrix
P_mat_dir = join(datadir, 'P_matrices/')

# constant parameters
H = 260
W = 344
num_joints = 13

# % Load camera matrices, camera centers, input and label files, and import CNN mode
P_mat_cam1 = np.load(join(P_mat_dir, 'P1.npy'))
P_mat_cam2 = np.load(join(P_mat_dir, 'P2.npy'))
P_mat_cam3 = np.load(join(P_mat_dir, 'P3.npy'))
P_mat_cam4 = np.load(join(P_mat_dir, 'P4.npy'))
P_mats = [P_mat_cam1, P_mat_cam2, P_mat_cam3, P_mat_cam4]
cameras_pos = np.load(join(P_mat_dir, 'camera_positions.npy'))


def get_all_joints(viconData, idx):
    viconMat = np.zeros([13, 3], dtype=viconData['XYZPOS']['head'].dtype)
    for i in range(0, len(viconData['XYZPOS'])):
        viconMat[i, :] = viconData['XYZPOS'][list(viconData['XYZPOS'])[i]][idx]
    return viconMat


def get_2Dcoords_and_heatmaps_label(vicon_xyz, ch_idx):
    # " From 3D label, get 2D label coordinates and heatmaps for selected camera "
    if ch_idx == 1:
        P_mat_cam = np.load(join(P_mat_dir, 'P1.npy'))
    elif ch_idx == 3:
        P_mat_cam = np.load(join(P_mat_dir, 'P2.npy'))
    elif ch_idx == 2:
        P_mat_cam = np.load(join(P_mat_dir, 'P3.npy'))
    elif ch_idx == 0:
        P_mat_cam = np.load(join(P_mat_dir, 'P4.npy'))
    # use homogeneous coordinates representation to project 3d XYZ coordinates to 2d UV pixel coordinates.
    vicon_xyz_homog = np.concatenate([vicon_xyz, np.ones([1, 13])], axis=0)
    coord_pix_homog = np.matmul(P_mat_cam, vicon_xyz_homog)
    coord_pix_homog_norm = coord_pix_homog / coord_pix_homog[-1]
    u = coord_pix_homog_norm[0]
    v = H - coord_pix_homog_norm[1]  # flip v coordinate to match the image direction
    # mask is used to make sure that pixel positions are in frame range.
    mask = np.ones(u.shape).astype(np.float32)
    mask[np.isnan(u)] = 0
    mask[np.isnan(v)] = 0
    mask[u > W] = 0
    mask[u <= 0] = 0
    mask[v > H] = 0
    mask[v <= 0] = 0
    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    return np.stack((v, u), axis=-1), mask


size = dataVicon['XYZPOS']['head'].shape[0]
dataType = dataVicon['XYZPOS']['head'].dtype
joints = {}
for ch in range(4):
    joints['ch' + str(ch)] = {}
    for key in dataVicon['XYZPOS']:
        joints['ch' + str(ch)][key] = np.empty((size, 2), dtype='uint16')
    for i in range(len(thz)):  # CHECK -1
        viconMat = get_all_joints(dataVicon, i)
        y_2d, gt_mask = get_2Dcoords_and_heatmaps_label(np.transpose(viconMat), ch)
        y_2d_float = y_2d.astype(np.uint16)
        for j in range(13):
            joints['ch' + str(ch)][list(joints['ch' + str(ch)])[j]][i, 1] = y_2d_float[j, 0].astype(np.uint16)
            joints['ch' + str(ch)][list(joints['ch' + str(ch)])[j]][i, 0] = y_2d_float[j, 1].astype(np.uint16)
    joints['ch' + str(ch)]['ts'] = thz

# %% Build  DVS+GT container
for ch in range(4):
    # Change dvs data to hpe
    container['data']['ch' + str(ch)]['hpe'] = container['data']['ch' + str(ch)].pop('dvs')
    # Add GT data
    container['data']['ch' + str(ch)]['hpe']['skeleton'] = {}
    container['data']['ch' + str(ch)]['hpe']['skeleton']['gt'] = joints['ch' + str(ch)]

# %% Plot joints vs t
import matplotlib.pyplot as plt

plt.close('all')
fig, ax = plt.subplots(2)
ch = 3  # choose cahnnel to plot
l = joints['ch' + str(ch)]['head'].shape[0]
for j in range(13):
    ax[0].plot(thz[0:l], joints['ch' + str(ch)][list(joints['ch' + str(ch)])[j]][:, 0], )
    ax[0].set_ylabel('x coordinate [px]', fontsize=12, labelpad=5)
    ax[1].plot(thz[0:l], joints['ch' + str(ch)][list(joints['ch' + str(ch)])[j]][:, 1])
    ax[1].set_ylabel('y coordinate [px]', fontsize=12, labelpad=5)

for i in range(2):
    ax[i].legend(list(joints['ch' + str(ch)]), loc='upper right', fontsize=11)
    ax[i].set_xlabel('time [sec]', fontsize=12, labelpad=-5)
    ax[i].grid()
fig.suptitle('Ground truth 2D position for ch' + str(ch) + ' camera', fontsize=18)

# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
plt.show()

# %% Plot events (RoI) + GT  (2 Roi values)
import matplotlib.pyplot as plt

plt.close('all')
fig, ax = plt.subplots(2, 2, sharex=True)

# choose joint and camera cahnnel to plot
ch = 3
joint = 'handL'
# set RoI values
RoI = [2, 4]
# set plot markers
plotMarkers = True


def findNearest(array, value):
    idx = np.searchsorted(array, value)  # side="left" param is the default
    if idx > 0 and (
                    idx == len(array) or
                    math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


# evs
t = container['data']['ch' + str(ch)]['hpe']['ts']
x = container['data']['ch' + str(ch)]['hpe']['x']
y = container['data']['ch' + str(ch)]['hpe']['y']
# extract events inside roi(t)
for idx, roi in enumerate(RoI):
    prev = -1
    xev = np.array([])
    yev = np.array([])
    tev = np.array([])
    for i in range(0, len(thz)):
        if i > 0:
            t2 = findNearest(t, thz[i]) - 1
        else:
            t2 = 0
        auxX = np.array(x[prev:t2])
        auxY = np.array(y[prev:t2])
        auxT = np.array(t[prev:t2])
        condX = np.logical_and(auxX > joints['ch' + str(ch)][joint][i, 0] - roi,
                               auxX < joints['ch' + str(ch)][joint][i, 0] + roi)
        condY = np.logical_and(auxY > joints['ch' + str(ch)][joint][i, 1] - roi,
                               auxY < joints['ch' + str(ch)][joint][i, 1] + roi)
        cond = np.logical_and(condX, condY)
        xev = np.append(xev, auxX[cond])
        yev = np.append(yev, auxY[cond])
        tev = np.append(tev, auxT[cond])
        prev = t2
    if plotMarkers:
        # With markers
        ax[0, idx].plot(tev, xev, color='red', marker=".")
        ax[1, idx].plot(tev, yev, color='red', marker=".")
        ax[0, idx].plot(thz, joints['ch' + str(ch)][joint][:, 0], marker="D", markersize=12)
        ax[1, idx].plot(thz, joints['ch' + str(ch)][joint][:, 1], marker="D", markersize=12)
    else:
        # Without markers
        ax[0, idx].plot(tev, xev, color='red')
        ax[1, idx].plot(tev, yev, color='red')
        ax[0, idx].plot(thz, joints['ch' + str(ch)][joint][:, 0])
        ax[1, idx].plot(thz, joints['ch' + str(ch)][joint][:, 1])
    ax[0, idx].set_ylabel('x coordinate [px]', fontsize=12, labelpad=5)
    ax[1, idx].set_ylabel('y coordinate [px]', fontsize=12, labelpad=5)
    ax[0, idx].set_title('RoI = ' + str(roi), fontsize=20)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# %% Plot events (RoI) + GT (FILTER)
import math
import matplotlib.pyplot as plt

plt.close('all')
fig, ax = plt.subplots(2, sharex=True)
# choose joint and camera cahnnel to plot
ch = 3
joint = 'handL'
# set RoI values
roi = 5
# set order of the polynomial used for filter
order = 3


def findNearest(array, value):
    idx = np.searchsorted(array, value)  # side="left" param is the default
    if idx > 0 and (
                    idx == len(array) or
                    math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


# evs
t = container['data']['ch' + str(ch)]['hpe']['ts']
x = container['data']['ch' + str(ch)]['hpe']['x']
y = container['data']['ch' + str(ch)]['hpe']['y']
# extract events inside roi(t)
prev = -1
xev = np.array([])
yev = np.array([])
tev = np.array([])
avg = 0
for i in range(0, len(thz)):
    if i > 0:
        t2 = findNearest(t, thz[i]) - 1
    else:
        t2 = 0
    auxX = np.array(x[prev:t2])
    auxY = np.array(y[prev:t2])
    auxT = np.array(t[prev:t2])
    condX = np.logical_and(auxX > joints['ch' + str(ch)][joint][i, 0] - roi,
                           auxX < joints['ch' + str(ch)][joint][i, 0] + roi)
    condY = np.logical_and(auxY > joints['ch' + str(ch)][joint][i, 1] - roi,
                           auxY < joints['ch' + str(ch)][joint][i, 1] + roi)
    cond = np.logical_and(condX, condY)
    xev = np.append(xev, auxX[cond])
    yev = np.append(yev, auxY[cond])
    tev = np.append(tev, auxT[cond])
    prev = t2
    avg += len(auxT[cond])
avg = int(avg / len(thz)) * 10
print('Average num evs per vicon sample = ', avg)
if avg % 2 == 0:
    avg += 1

# Without markers
# ax[0].plot(tev, xev, color='red')
# # ax[1].plot(tev, yev, color='red')
# # GT
# ax[0].plot(thz, joints['ch'+str(ch)][joint][:,0])
# ax[0].set_ylabel('x coordinate [px]', fontsize=12, labelpad=5)
# ax[1].plot(thz, joints['ch'+str(ch)][joint][:,1])
# ax[1].set_ylabel('y coordinate [px]', fontsize=12, labelpad=5)
# # With big markers
# ax[0].plot(tev, xev, color='red',marker = ".")
# ax[1].plot(tev, yev, color='red',marker = ".")
# # GT
# ax[0].plot(thz, joints['ch'+str(ch)][joint][:,0],marker = "D", markersize=12)
# ax[0].set_ylabel('x coordinate [px]', fontsize=12, labelpad=5)
# ax[1].plot(thz, joints['ch'+str(ch)][joint][:,1],marker = "D", markersize=12)
# ax[1].set_ylabel('y coordinate [px]', fontsize=12, labelpad=5)


#  Filter
from scipy.signal import savgol_filter  # Savitzky-Golay filter

# x-axis
ax[0].plot(tev, xev, color='red', marker=".", label='raw events')
ax[0].plot(thz, joints['ch' + str(ch)][joint][:, 0], marker="D", markersize=12, label='GT')
ax[0].set_ylabel('x coordinate [px]', fontsize=12, labelpad=5)
xS = savgol_filter(xev, avg, order)
ax[0].plot(tev, xS, color='blue', marker="o", label='filtered events')

# y-axis
ax[1].plot(tev, yev, color='red', marker=".")
ax[1].plot(thz, joints['ch' + str(ch)][joint][:, 1], marker="D", markersize=12)
ax[1].set_ylabel('y coordinate [px]', fontsize=12, labelpad=5)
yS = savgol_filter(yev, avg, order)
ax[1].plot(tev, yS, color='blue', marker="o")

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

fig.legend(lines, labels, loc=1, prop={'size': 16})
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# %% Plot GT 10Hz
import matplotlib.pyplot as plt

plt.close('all')
fig, ax = plt.subplots(2, sharex=True)
# choose joint and camera cahnnel to plot
ch = 3
joint = 'handL'

f = 10
T = int(100 / f)
gtf = joints['ch' + str(ch)][joint][::T, :]
tf = thz[::T]

# x-axis
ax[0].plot(tf, gtf[:, 0], color='red', marker="o", markersize=12, label='GT')
ax[0].plot(thz, joints['ch' + str(ch)][joint][:, 0], marker="*", markersize=8, label='sampled GT')
ax[0].set_ylabel('x coordinate [px]', fontsize=14, labelpad=5)
ax[0].set_title('Left hand joint  - Sampling freq = ' + str(f) + ' Hz', fontsize=18)
# inset axes
axins = ax[0].inset_axes([0.5, 0.1, 0.45, 0.8])
# sub region of the original image
x1, x2, y1, y2 = 22, 25, 168, 215
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
axins.plot(tf, gtf[:, 0], color='red', marker="o", markersize=12)
axins.plot(thz, joints['ch' + str(ch)][joint][:, 0], marker="*", markersize=8)
axins.set_facecolor('gainsboro')
ax[0].indicate_inset_zoom(axins, edgecolor="black", label='_nolegend_')

# y-axis
ax[1].plot(tf, gtf[:, 1], color='red', marker="o", markersize=12)
ax[1].plot(thz, joints['ch' + str(ch)][joint][:, 1], marker="*", markersize=8)
ax[1].set_ylabel('y coordinate [px]', fontsize=14, labelpad=5)
ax[1].set_xlabel('time [sec]', fontsize=14)
axins = ax[1].inset_axes([0.5, 0.15, 0.45, 0.8])
y1, y2 = 115, 150
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
axins.plot(tf, gtf[:, 1], color='red', marker="o", markersize=12)
axins.plot(thz, joints['ch' + str(ch)][joint][:, 1], marker="*", markersize=8)
axins.set_facecolor('gainsboro')
ax[1].indicate_inset_zoom(axins, edgecolor="black", label='_nolegend_')

# global legend
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc=1, prop={'size': 16})
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
plt.tight_layout()

# %% Generate a second skeleton with noise to see if comparisson works
import copy

noisySkt = copy.deepcopy(joints)
for ch in range(4):
    for key in noisySkt['ch' + str(ch)]:
        if key != 'ts':
            noise = np.int_(np.random.normal(0, 1, noisySkt['ch' + str(ch)][key].shape))
            noisySkt['ch' + str(ch)][key] = noisySkt['ch' + str(ch)][key] + noise

for ch in range(4):
    container['data']['ch' + str(ch)]['hpe']['skeleton']['test'] = noisySkt['ch' + str(ch)]

# %% Container with just one channel
contCh3 = {}
contCh3['info'] = container['info']
contCh3['data'] = {}
contCh3['data']['ch3'] = container['data']['ch3']

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

# %% Visualize data
app.setData(container)
# app.setData(contCh3)


# %% Export to yarp (only DVS)
from bimvee import exportIitYarp

datafile = 'S{}_{}_{}'.format(subj, sess, mov)

# exportIitYarp.exportIitYarp(container, 
#               exportFilePath= '/home/fdipietro/hpe-data/yarp/'+datafile, 
#               pathForPlayback= '/home/fdipietro/hpe-data/yarp/'+datafile,
#               protectedWrite = False)
exportIitYarp.exportIitYarp(container,
                            exportFilePath='/home/fdipietro/hpe-data/yarp/' + datafile,
                            protectedWrite=False)
