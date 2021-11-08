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
vel= np.empty((len(lines),3))
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
 

ax2.legend(fontsize=18, loc = 'lower right')

plt.show()

# %% First plots


# fig = plt.figure(figsize=(2048/my_dpi, 1600/my_dpi), dpi=my_dpi)
# ax = plt.subplot(111)
# ax.plot(data[:,0],data[:,j*2-1], marker = ".", label =r'$x_{tracked}$',linestyle = 'None')
# ax.plot(data[:,0],data[:,j*2], marker = ".", label =r'$y_{tracked}$',linestyle = 'None')
# plt.xlabel('time [sec]', fontsize=18, labelpad=5)
# plt.ylabel('x/y coordinate [px]', fontsize=18, labelpad=5)
# fig.suptitle('Joint tracker output - S_111 - '+p[j], fontsize=22, y=0.95)
# plt.tick_params(axis='both', which='major', labelsize=14)
# if(plotGT):
#     # ax.plot(t_gt,gt[:,j-1,0],color='tab:blue', alpha=0.25, label =r'$x_{GT}$')
#     # ax.plot(t_gt,gt[:,j-1,1],color='tab:orange', alpha=0.25, label =r'$y_{GT}$')
#     ax.plot(t_gt,x,color='tab:blue', alpha=0.25, label =r'$x_{GT}$')
#     ax.plot(t_gt,y,color='tab:orange', alpha=0.25, label =r'$y_{GT}$')
#     ax.plot(t_gt,dxf,color='blue', alpha=0.25, label =r'$x_{GT}$')
#     ax.plot(t_gt,dyf,color='orange', alpha=0.25, label =r'$y_{GT}$')
# ax.legend(fontsize=18, loc = 'lower right')
# plt.show()

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
    # ax2.plot(t_gt,15*dx+200,color='blue', alpha=0.15, label =r'$x_{GT}$')
    ax2.plot(t_gt,dxf,color='blue', alpha=0.25, label =r'$dx_{GT}$', linestyle = '--')
    ax2.plot(t_gt,np.ones(np.shape(t_gt))*mx,color='blue', alpha=0.1, linestyle = '--')
    # ax2.plot(t_gt,15*dy+100,color='orange', alpha=0.25, label =r'$y_{GT}$')
    ax2.plot(t_gt,dyf,color='orange', alpha=0.25, label =r'$dy_{GT}$', linestyle = '--')
    ax2.plot(t_gt,np.ones(np.shape(t_gt))*my,color='red', alpha=0.1, linestyle = '--')
    
ax2.plot(aux[:,0],aux[:,j*2-1], marker = 'x', markersize = 8,linestyle = 'None', color='blue')
ax2.plot(aux[:,0],aux[:,j*2], marker = 'x', markersize = 8,linestyle = 'None', color='red')


ax2.plot(vel[:,0],vel[:,1], label =r'$n_{evs}$',linestyle = 'None', alpha=0.25, marker = ".")
ax2.plot(vel[:,0],vel[:,2], label =r'$n_{roi}$',linestyle = 'None', alpha=0.25, marker = ".")

# ax2.plot(vel[:,0],0.05*vel[:,1]+mx, label =r'$n_{evs}$',linestyle = '-', alpha=0.25, marker = ".")
# ax2.plot(vel[:,0],0.05*vel[:,2]+my, label =r'$n_{roi}$',linestyle = '-', alpha=0.25, marker = ".")


# ax2.plot(tv,1000*vx+220, marker = ".", label =r'$dx$' ,linestyle = 'None')
# ax2.plot(tv,1000*vy+110, marker = ".", label =r'$dy$',linestyle = 'None')

ax2.legend(fontsize=18, loc = 'lower right')

plt.show()

# fig3 = plt.figure(figsize=(2048/my_dpi, 1600/my_dpi), dpi=my_dpi)
# ax3 = plt.subplot(111)
# # ax3.plot(vy,vx, marker = ".", label =r'$x_{tracked}$',linestyle = 'None')
# ax3.scatter(vx,vy)
# ax3.plot([0, avx], [0, avy])
# # ax3.arrow(0,0, avx*0.92,avy*0.92, head_width = 0.02)
# # ax3.arrow(0,0, avx/2,avy/2, head_width = 0.02)

# xc = avx/2
# yc = avy/2
# r = np.sqrt((avx/2)**2 + (avy/2)**2)

# theta_fit = np.linspace(-np.pi, np.pi, 180)
# x_fit = xc + r*np.cos(theta_fit)
# y_fit = yc + r*np.sin(theta_fit)
# ax3.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
# ax3.set_aspect('equal')


# %% Twinx figure
# fig, [ax, bx] = plt.subplots(2)
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.85)


dxf = savgol_filter(dx, 13, order)
dyf = savgol_filter(dy, 13, order)

twin1 = ax.twinx()
twin2 = ax.twinx()
twin2.spines['right'].set_position(("axes", 1.1))
ax.set_xlim([0.8, 4.2])

# x component
p1, = ax.plot(data[:,0],data[:,j*2-1], marker = ".", label =r'$x_{tracked}$',linestyle = 'None')
p12, = ax.plot(t_gt,gt[:,j-1,0],color='tab:blue', alpha=0.3, label =r'$x_{GT}$')
p2, = twin1.plot(t_gt,dxf,color='red', alpha=0.25, label =r'$dx_{GT}/dt$', linestyle = '--')
p22, = twin1.plot(t_gt,np.zeros(np.shape(t_gt)),color='gray', alpha=0.05, linestyle = '--')
p3, = twin2.plot(vel[:,0],vel[:,1], label =r'$n_{evs}$',linestyle = 'None', alpha=0.2, marker = ".", color='tab:green')

ax.set_xlabel("Time", fontsize=18)
ax.set_ylabel("x- pose [px]", fontsize=18)
twin1.set_ylabel("Velocity [px/sec]", fontsize=18)
twin2.set_ylabel("Events", fontsize=18)
ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())
tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), labelsize=14, **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), labelsize=14, **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), labelsize=14, **tkw)
ax.tick_params(axis='x', **tkw)
ax.legend(handles=[p1, p12, p2, p3], fontsize=18, loc = 'upper left')
fig.suptitle('Joint tracker output - S_111 - '+p[j], fontsize=22, y=0.95)


# twin3 = bx.twinx()
# twin4 = bx.twinx()
# twin4.spines['right'].set_position(("axes", 1.1))
# bx.set_xlim([0.8, 4.2])


# q1, = bx.plot(data[:,0],data[:,j*2],color='tab:orange', marker = ".", label =r'$y_{tracked}$',linestyle = 'None')
# q12, = bx.plot(t_gt,gt[:,j-1,1],color='tab:orange', alpha=0.3, label =r'$y_{GT}$')
# q2, = twin3.plot(t_gt,dyf,color='red', alpha=0.25, label =r'$dx_{GT}$', linestyle = '--')
# q22, = twin3.plot(t_gt,np.zeros(np.shape(t_gt)),color='gray', alpha=0.05, linestyle = '--')
# q3, = twin4.plot(vel[:,0],vel[:,1], label =r'$n_{evs}$',linestyle = 'None', alpha=0.1, marker = ".", color='tab:green')

# bx.set_xlabel("Time", fontsize=18)
# bx.set_ylabel("y- pose [px]", fontsize=18)
# twin3.set_ylabel("Velocity [px/sec]", fontsize=18)
# twin4.set_ylabel("Events", fontsize=18)
# bx.yaxis.label.set_color(q1.get_color())
# twin3.yaxis.label.set_color(q2.get_color())
# twin4.yaxis.label.set_color(q3.get_color())
# tkw = dict(size=4, width=1.5)
# bx.tick_params(axis='y', colors=q1.get_color(), labelsize=14, **tkw)
# twin3.tick_params(axis='y', colors=q2.get_color(), labelsize=14, **tkw)
# twin4.tick_params(axis='y', colors=q3.get_color(), labelsize=14, **tkw)
# bx.tick_params(axis='x', **tkw)
# bx.legend(handles=[q1, q12, q2, q3], fontsize=18, loc = 'upper left')


plt.show()
   
# %% Twinx figure NO vel
# fig, [ax, bx] = plt.subplots(2)
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.85)

twin1 = ax.twinx()
ax.set_xlim([0.8, 4.2])

# x component
p1, = ax.plot(data[:,0],data[:,j*2-1], marker = ".", label =r'$x_{tracked}$',linestyle = 'None')
p12, = ax.plot(t_gt,gt[:,j-1,0],color='tab:blue', alpha=0.3, label =r'$x_{GT}$')
# p2, = twin1.plot(t_gt,dxf,color='red', alpha=0.25, label =r'$dx_{GT}/dt$', linestyle = '--')
# p22, = twin1.plot(t_gt,np.zeros(np.shape(t_gt)),color='gray', alpha=0.05, linestyle = '--')
p2, = twin1.plot(vel[:,0],vel[:,1], label =r'$n_{evs}$',linestyle = 'None', alpha=0.2, marker = ".", color='tab:green')

ax.set_xlabel("Time", fontsize=18)
ax.set_ylabel("x- pose [px]", fontsize=18)
# twin1.set_ylabel("Velocity [px/sec]", fontsize=18)
twin1.set_ylabel("Events", fontsize=18)
ax.yaxis.label.set_color(p1.get_color())
# twin1.yaxis.label.set_color(p2.get_color())
twin1.yaxis.label.set_color(p3.get_color())
tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), labelsize=14, **tkw)
# twin1.tick_params(axis='y', colors=p2.get_color(), labelsize=14, **tkw)
twin1.tick_params(axis='y', colors=p3.get_color(), labelsize=14, **tkw)
ax.tick_params(axis='x', **tkw)
# ax.legend(handles=[p1, p12, p2, p3], fontsize=18, loc = 'upper left')
ax.legend(handles=[p1, p12, p2], fontsize=18, loc = 'upper left')
fig.suptitle('Joint tracker output - S_111 - '+p[j], fontsize=22, y=0.95)



plt.show()