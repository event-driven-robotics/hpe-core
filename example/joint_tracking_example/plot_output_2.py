#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:01:04 2021

@author: fdipietro
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:20:17 2021

@author: fdipietro
"""
# %% Load data
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load trakcing output
prefix = '/home/fdipietro/projects/hpe-core/example/joint_tracking_example/build/' 
filename = 'output.txt'

# Load ground-truth
lines = tuple(open(prefix+filename, 'r'))
data= np.empty((len(lines),27))
for i in range(len(lines)):
    data[i,:] = np.array(lines[i].split())
# selected recording
subj, sess, mov = 1, 1, 1
recording = 'S{}_{}_{}'.format(subj, sess, mov)
cam = str(3)
gt2Dfile = '/home/fdipietro/hpe-data/gt2D/' + recording +'.pkl'
with open(gt2Dfile, 'rb') as f:
    dataGT = pickle.load(f)
# Build full GT (for plots)
L = len(dataGT['ch3']['ts'])
gt = np.empty([L,13,2])
for j in range(L):
    i=0
    for key, value in dataGT['ch3'].items():
        if(key!='ts'):
            gt[j,i,:] = value[j]
            i=i+1
t_gt = np.array(dataGT['ch3']['ts'])


# Load aux output
aux = 'aux_out.txt'
lines = tuple(open(prefix+aux, 'r'))
aux= np.empty((len(lines),27))
for i in range(len(lines)):
    aux[i,:] = np.array(lines[i].split())
    
    
# # Load vel output
velFile = 'vel_out.txt'
lines = tuple(open(prefix+velFile, 'r'))
# vel= np.empty((len(lines),3))
# for i in range(len(lines)):
#     vel[i,:] = np.array(lines[i].split(","))
vel= np.empty((len(lines),2))
for i in range(len(lines)-1):
    vel[i,:] = np.array(lines[i].split())
    
# #select non nan
# lim = 10
# vx = vel[np.bitwise_and(abs(vel[:,1]) < lim, abs(vel[:,2]) < lim),1]
# vy = vel[np.bitwise_and(abs(vel[:,1]) < lim, abs(vel[:,2]) < lim),2]
# tv = vel[np.bitwise_and(abs(vel[:,1]) < lim, abs(vel[:,2]) < lim),0]

# tinf = 2.4105
# tsup = tinf + 0.1 *5
# # select one period
# vx = vel[np.bitwise_and(vel[:,0] > tinf, vel[:,0] < tsup),1]
# vy = vel[np.bitwise_and(vel[:,0] > tinf, vel[:,0] < tsup),2]
# tv = vel[np.bitwise_and(vel[:,0] > tinf, vel[:,0] < tsup),0]

# avx = np.mean(vx)
# avy = np.mean(vy)
# L = len(vx)
# for i in range(L):
#     vx = np.append(vx, 0)
#     vy = np.append(vy, 0)
#     tv = np.append(tv, 0)





# Plots
plt.close('all')
plotGT = True

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

j = DHP19_BODY_PARTS['handL']
p = dict(zip(DHP19_BODY_PARTS.values(),DHP19_BODY_PARTS.keys()))

# Compute deivatives
from numpy import diff
dt = 0.01
x = gt[:,j-1,0]
y = gt[:,j-1,1]
mx = np.mean(x)
my = np.mean(y)
Ax = abs(np.max(x)-np.min(x))
Ay = abs(np.max(y)-np.min(y))
# dx = diff(x)/dt
# dy = diff(y)/dt
dx = np.gradient(x, edge_order=2)
dy = np.gradient(y)
Adx = abs(np.max(dx)-np.min(dx))
Ady = abs(np.max(dy)-np.min(dy))

from scipy.signal import savgol_filter

order = 3
dxf = Ax/Adx*savgol_filter(dx, 13, order)+mx
dyf = Ay/Ady*savgol_filter(dy, 13, order)+my

# dxf = np.polyfit(dx, t_gt, 6)

my_dpi = 96

# %% SIE plots
fig2 = plt.figure(figsize=(2048/my_dpi, 1600/my_dpi), dpi=my_dpi)
ax2 = plt.subplot(111)
ax2.plot(data[:,0],data[:,j*2-1], marker = ".", label =r'$x_{tracked}$',linestyle = 'None')
ax2.plot(data[:,0],data[:,j*2], marker = ".", label =r'$y_{tracked}$',linestyle = 'None')
ax2.set_xlim([0.8, 4.2])
ax2.set_ylim([0, 270])
plt.xlabel('time [sec]', fontsize=18, labelpad=5)
plt.ylabel('x/y coordinate [px]', fontsize=18, labelpad=5)
fig2.suptitle('Joint tracker output - S_111 - '+p[j], fontsize=22, y=0.95)
plt.tick_params(axis='both', which='major', labelsize=14)
if(plotGT):
    ax2.plot(t_gt,gt[:,j-1,0],color='tab:blue', alpha=0.3, label =r'$x_{GT}$')
    ax2.plot(t_gt,gt[:,j-1,1],color='tab:orange', alpha=0.3, label =r'$y_{GT}$')


ax2.plot(vel[:,0],(vel[:,1])*100, label =r'$corr_{proj-new}$',linestyle = 'solid', alpha=0.25, marker = ".")

ax2.legend(fontsize=18, loc = 'upper left')

dx = np.ediff1d(data[:,j*2-1])
dy = np.ediff1d(data[:,j*2])
tdx = data[:-1,0]
tdxx = np.array([data[1,0], data[-1,0]])

# # fig3 = plt.figure(figsize=(2048/my_dpi, 1600/my_dpi), dpi=my_dpi)
# ax3= plt.subplot(212)
# ax3.plot(tdx, dx, marker = ".", label =r'$\Delta x_{tracked}$')
# ax3.plot(tdx, dy, marker = ".", label =r'$\Delta y_{tracked}$')
# ax3.plot(tdxx, np.ones(len(tdxx)), linestyle = 'dotted', color='gray')
# ax3.plot(tdxx, -np.ones(len(tdxx)), linestyle = 'dotted', color='gray')
# ax3.set_xlim([0.8, 4.2])
# ax3.set_ylim([-1.25, 1.25])
# plt.xlabel('time [sec]', fontsize=18, labelpad=5)
# plt.ylabel(r'$\Delta$x coordinate [px]', fontsize=18, labelpad=5)
# # fig3.suptitle('Joint tracker output - S_111 - '+p[j], fontsize=22, y=0.95)
# plt.tick_params(axis='both', which='major', labelsize=14)
 
# ax3.legend(fontsize=18, loc = 'lower right')
plt.show()

