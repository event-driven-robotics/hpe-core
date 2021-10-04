#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:20:17 2021

@author: fdipietro
"""
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import csv
# import math 
import pickle

plt.close('all')

# CONVERT OBJECT TRACKING pose.txt INTO CSV FILE
#prefix = '/home/fdipietro/Desktop/' 
prefix = '/home/fdipietro/projects/hpe-core/example/joint_tracking_example/build/' 
#filename = 'pose_slow2.txt'
filename = 'output.txt'

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
    
# build full GT (for plots)
L = len(dataGT['ch3']['ts'])
gt = np.empty([L,13,2])
for j in range(L):
    i=0
    for key, value in dataGT['ch3'].items():
        if(key!='ts'):
            gt[j,i,:] = value[j]
            i=i+1
t_gt = np.array(dataGT['ch3']['ts'])


# PLots
plt.close()

plotGT = True

my_dpi = 96
fig = plt.figure(figsize=(2048/my_dpi, 1600/my_dpi), dpi=my_dpi)
ax = plt.subplot(111)
# coord = 1
j = 9
ax.plot(data[:,0],data[:,j*2-1], marker = ".", label =r'$x_{tracked}$',linestyle = 'None')
ax.plot(data[:,0],data[:,j*2], marker = ".", label =r'$y_{tracked}$',linestyle = 'None')
# ax.set_xlim([0.8, 4.2])
# ax.set_ylim([30, 250])
plt.xlabel('time [sec]', fontsize=18, labelpad=5)
plt.ylabel('x/y coordinate [px]', fontsize=18, labelpad=5)

fig.suptitle('Joint tracker output - S_111 - left hand', fontsize=22, y=0.95)
plt.tick_params(axis='both', which='major', labelsize=14)
if(plotGT):
    ax.plot(t_gt,gt[:,j-1,0],color='tab:blue', alpha=0.25, label =r'$x_{GT}$')
    ax.plot(t_gt,gt[:,j-1,1],color='tab:orange', alpha=0.25, label =r'$y_{GT}$')
ax.legend(fontsize=18, loc = 'lower right')
plt.show()

fig2 = plt.figure(figsize=(2048/my_dpi, 1600/my_dpi), dpi=my_dpi)
ax2 = plt.subplot(111)
# coord = 1
j = 9
ax2.plot(data[:,0],data[:,j*2-1], marker = ".", label =r'$x_{tracked}$',linestyle = 'None')
ax2.plot(data[:,0],data[:,j*2], marker = ".", label =r'$y_{tracked}$',linestyle = 'None')
ax2.set_xlim([0.8, 4.2])
ax2.set_ylim([30, 250])
plt.xlabel('time [sec]', fontsize=18, labelpad=5)
plt.ylabel('x/y coordinate [px]', fontsize=18, labelpad=5)

fig2.suptitle('Joint tracker output - S_111 - left hand', fontsize=22, y=0.95)
plt.tick_params(axis='both', which='major', labelsize=14)
if(plotGT):
    ax2.plot(t_gt,gt[:,j-1,0],color='tab:blue', alpha=0.25, label =r'$x_{GT}$')
    ax2.plot(t_gt,gt[:,j-1,1],color='tab:orange', alpha=0.25, label =r'$y_{GT}$')
ax2.legend(fontsize=18, loc = 'lower right')
plt.show()
