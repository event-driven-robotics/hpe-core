#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:01:04 2021

@author: fdipietro
"""

# %% Load data
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load trakcing output
prefix = '/home/fdipietro/projects/hpe-core/example/joint_tracking_example/build/' 
filename = 'output.txt'
lines = tuple(open(prefix+filename, 'r'))
data= np.empty((len(lines),27))
for i in range(len(lines)):
    data[i,:] = np.array(lines[i].split())

# Load ground-truth
subj, sess, mov = 1, 1, 1 # Selected recording
recording = 'S{}_{}_{}'.format(subj, sess, mov)
cam = str(3)
gt2Dfile = '/home/fdipietro/hpe-data/gt2D/' + recording +'.pkl'
with open(gt2Dfile, 'rb') as f:
    dataGT = pickle.load(f)
# Build full 100Hz ground-truth (for plots)
L = len(dataGT['ch3']['ts'])
gt = np.empty([L,13,2])
for j in range(L):
    i=0
    for key, value in dataGT['ch3'].items():
        if(key!='ts'):
            gt[j,i,:] = value[j]
            i=i+1
t_gt = np.array(dataGT['ch3']['ts'])

# Load detections (aux output in tracking app)
aux = 'aux_out.txt'
lines = tuple(open(prefix+aux, 'r'))
aux= np.empty((len(lines),27))
for i in range(len(lines)):
    aux[i,:] = np.array(lines[i].split())
    
# Load velocities estimations (vel output in tracking app)
vel = 'vel_out.txt'
lines = tuple(open(prefix+vel, 'r'))
vel= np.empty((len(lines),3))
for i in range(len(lines)):
    if(i!=0):
        vel[i,:] = np.array(lines[i].split())
    
# Get index for selected joint
DHP19_BODY_PARTS = { # map from body parts to indices for dhp19
    'head': 1,
    'shoulderR': 2,
    'shoulderL': 3,
    'elbowR': 4,
    'elbowL': 5,
    'hipL': 6,
    'hipR': 7,
    'handR': 8,
    'handL': 9,
    'kneeR': 10,
    'kneeL': 11,
    'footR': 12,
    'footL': 13
}
j = DHP19_BODY_PARTS['handL'] # selected joint
p = dict(zip(DHP19_BODY_PARTS.values(),DHP19_BODY_PARTS.keys()))

plt.close('all')
my_dpi = 96

# %% Plot 1: Tracking a ground-truth v time
plotGT = True
# detections
x10 = aux[:,j*2-1]
y10 = aux[:,j*2]
t10 = aux[:,0]
#tracking
st = 1
xTR = data[st:,j*2-1]
yTR = data[st::,j*2]
t = data[st::,0]
idx = np.logical_and(np.logical_not(np.isnan(xTR)), np.logical_not(np.isnan(yTR)))
xTR = xTR[idx]
yTR = yTR[idx]
t = t[idx]
xOS = np.interp(t, t_gt, gt[:,j-1,0])
yOS = np.interp(t, t_gt, gt[:,j-1,1])


fig1 = plt.figure(figsize=(2048/my_dpi, 900/my_dpi), dpi=my_dpi)
ax1 = plt.subplot(111)
fig1.tight_layout(pad=5)
ax1.plot(t,xTR, marker = ".", label =r'$x_{tracked}$',linestyle = 'None', alpha=1)
ax1.plot(t,yTR, marker = ".", label =r'$y_{tracked}$',linestyle = 'None', alpha=1)
ax1.set_xlim([1, t[-1]/2])
ax1.set_ylim([min(min(xTR),min(yTR))*0.6, max(max(xTR),max(yTR))*1.4])

plt.xlabel('time [sec]', fontsize=22, labelpad=5)
plt.ylabel('x/y coordinate [px]', fontsize=22, labelpad=5)
fig1.suptitle('Joint tracker output - '+recording+' - '+p[j], fontsize=28, y=0.97)
plt.tick_params(axis='both', which='major', labelsize=18)
if(plotGT):
    ax1.plot(t,xOS,color='tab:blue', alpha=0.3, label =r'$x_{GT}$')
    ax1.plot(t,yOS,color='tab:orange', alpha=0.3, label =r'$y_{GT}$')
ax1.legend(fontsize=18, loc = 'upper right')
plt.show()
    
# %%Plot 2: Observation model - velocities circle
    
# Compute deivatives
dt = 0.1
dx = np.gradient(x10)/dt
dy = np.gradient(y10)/dt

t0 = 3.7 # starting time to plot
tf = t0+dt

# get velocities in the selected interval
idx1 = np.argwhere(np.logical_and(t10 > t0, t10 < tf))
tdx1 = t10[idx1]
dx1 = dx[idx1]
dy1 = dy[idx1]

idx2 = np.argwhere(np.logical_and(vel[:,0] > t0, vel[:,0] < tf))
auxt = vel[:,0]
auxdx = vel[:,1]
auxdy = vel[:,2]
tdx2 = auxt[idx2]
dx2 = auxdx[idx2]
dy2 = auxdy[idx2]

fig2 = plt.figure(figsize=(2048/my_dpi, 1600/my_dpi), dpi=my_dpi)
ax2= plt.subplot(111)
ax2.plot(dx2,dy2, marker = ".", label =r'$d\hat x/dt$',linestyle = 'None', color='black', markersize = 2)
# determine circle based on ground-truth veolicty
cx = np.average(dx1)
cy = np.average(dy1)
rad = np.sqrt(cx*cx+cy*cy)/2
circle1 = plt.Circle((cx/2, cy/2), rad, color='r', linestyle='--', fill = False, linewidth=3)
plt.gca().add_patch(circle1)
plt.gca().set_aspect('equal')
plt.arrow(0, 0, cx, cy, head_width=5, length_includes_head = True, width=1.5, color='r', label =r'$dx/dt$')
plt.xlabel(r'$v_x$', fontsize=22, labelpad=5)
plt.ylabel(r'$v_y$', fontsize=22, labelpad=5)
plt.show()