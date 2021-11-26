#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fdipietro
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# map from body parts to indices for dhp19
DHP19_BODY_PARTS = {
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

# selected recording
subj, sess, mov = 1, 1, 1
# select joint
j = DHP19_BODY_PARTS['handL']
p = dict(zip(DHP19_BODY_PARTS.values(),DHP19_BODY_PARTS.keys()))

# Load trakcing output
prefix = '/home/fdipietro/projects/hpe-core/example/joint_tracking_example/build/' 
filename = 'output.txt'

# Load ground-truth
lines = tuple(open(prefix+filename, 'r'))
data= np.empty((len(lines),27))
for i in range(len(lines)):
    data[i,:] = np.array(lines[i].split())

recording = 'S{}_{}_{}'.format(subj, sess, mov)
cam = str(3)
gt2Dfile = '/home/fdipietro/hpe-data/gt2D/' + recording +'.pkl'
with open(gt2Dfile, 'rb') as f:
    dataGT = pickle.load(f)
# Build full GT (for plots)
L = len(dataGT['ch3']['ts'])
gt = np.empty([L,13,2])
for k in range(L):
    i=0
    for key, value in dataGT['ch3'].items():
        if(key!='ts'):
            gt[k,i,:] = value[k]
            i=i+1
t_gt = np.array(dataGT['ch3']['ts'])


# Interpolate GT to tracking output
import math 
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

# calculate nrmse
nrmse = np.zeros(2)
from sklearn.metrics import mean_squared_error
mse = np.zeros(2)
mse[0] = mean_squared_error(xOS, xTR)
nrmse[0] = math.sqrt(mse[0])
mse[1] = mean_squared_error(yOS, yTR)
nrmse[1] = math.sqrt(mse[1])
Gnrmse = np.amax(nrmse)
print('============== sklearn ===================')
print("nRMSE(x) = ", nrmse[0] )
print("nRMSE(y) = ", nrmse[1] )
print("nRMSE = ",Gnrmse)
print('==========================================')
ex = str("%.2f%%" % round(nrmse[0],2))
ey = str("%.2f%%" % round(nrmse[1],2))


# Plots
plt.close('all')
plotGT = True
my_dpi = 96
fig = plt.figure(figsize=(2048/my_dpi, 900/my_dpi), dpi=my_dpi)
ax = plt.subplot(111)
fig.tight_layout(pad=5)
ax.plot(t,xTR, marker = ".", label =r'$x_{tracked}\,,\,e_x=$'+ex,linestyle = 'None')
ax.plot(t,yTR, marker = ".", label =r'$y_{tracked}\,,\,e_y=$'+ey,linestyle = 'None')
# ax2.set_xlim([0.8, 4.2])
# ax2.set_ylim([0, 270])
ax.set_xlim([1, t[-1]/2])
ax.set_ylim([min(min(xTR),min(yTR))*0.6, max(max(xTR),max(yTR))*1.4])
plt.xlabel('time [sec]', fontsize=22, labelpad=5)
plt.ylabel('x/y coordinate [px]', fontsize=22, labelpad=5)
fig.suptitle('Joint tracker output - '+recording+' - '+p[j], fontsize=28, y=0.97)
plt.tick_params(axis='both', which='major', labelsize=18)
if(plotGT):
    ax.plot(t,xOS,color='tab:blue', alpha=0.3, label =r'$x_{GT}$')
    ax.plot(t,yOS,color='tab:orange', alpha=0.3, label =r'$y_{GT}$')
ax.legend(fontsize=22, loc = 'upper left')
plt.show()

