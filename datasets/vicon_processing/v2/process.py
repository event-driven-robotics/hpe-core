# okay first off lets load in all the data what do we need
# event data stream
# results file with transform and delay
# vicon file with marker positions 3D and camera marker positions 3D
# we also need the camera calibration
#
#%% SET-UP ARGUMENTS ---------------------------------
import sys
sys.path.append('/home/aglover-iit.local/code/hpe-core/datasets/vicon_processing/v2')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from bimvee.importIitYarp import importIitYarp
import c3d
import helpers
import importlib
import math

importlib.reload(helpers)
 
sys.argv = ['Interactive-1.ipynb', 
            '--events_path', '/home/aglover-iit.local/data/move-iit-hpe-subset1/P1/tennis_f1/atis-s/',
            '--c3d_path', '/home/aglover-iit.local/data/move-iit-hpe-subset1/P1/tennis_f1.c3d',
            '--calib_path', '/home/aglover-iit.local/data/move-iit-hpe-subset1/P1/calib-s.txt']

parser = argparse.ArgumentParser(prog='Extrinsic Calibration of Markers to Camera Focal Point')
parser.add_argument('--events_path', 
                    required=True, 
                    help='path to the events dataset')
parser.add_argument('--c3d_path', 
                    required=True, 
                    help='path to the 3D marker/joint positions')
parser.add_argument('--fps', 
                    default=25.0,
                    help='path to the 3D marker/joint positions')
parser.add_argument('--calib_path', 
                    required=True, 
                    help='path to the camera calibration')
args = parser.parse_args()

period = 1.0 / args.fps

np.set_printoptions(precision=3,suppress=True)

#%% ---------------------------------
#IMPORT EVENT DATA
v_data = importIitYarp(filePathOrName=args.events_path, zeroTimestamp=False)
e_ts = v_data['data']['left']['dvs']['ts']
e_us = v_data['data']['left']['dvs']['x']
e_vs = v_data['data']['left']['dvs']['y']

#%% ---------------------------------
# IMPORT C3D DATA
c3d_data = c3d.Reader(open(args.c3d_path, 'rb'))
points_3d = {}
for i, points, analog in c3d_data.read_frames():
    points_3d[i] = points

marker_t = np.linspace(0.0, c3d_data.frame_count / c3d_data.point_rate, 
                            c3d_data.frame_count, endpoint=False)

#%% ---------------------------------
#IMPORT CALIBRATION DATA
calib = np.genfromtxt(args.calib_path, delimiter=" ", skip_header=1, dtype=object)

calib_dict = {}
keys = calib[:, 0].astype(str)
vals = calib[:, 1].astype(float)
calib_dict = {
    key: value for key, value in zip(keys, vals)
}
cam_res = np.int64([calib_dict['h'], calib_dict['w']])
cam_res_rgb = np.append(cam_res, 3)
# intrinsic
K = np.array([
    [calib_dict['fx'], 0.0, calib_dict['cx']],
    [0.0, calib_dict['fy'], calib_dict['cy']],
    [0.0, 0.0, 1.0]
])
# camera distortion
D = np.array([calib_dict['k1'], calib_dict['k2'], calib_dict['p1'], calib_dict['p2']])

#%% ---------------------------------
#VISUALISE EVENT DATA



#range(ts[0]+1, ts[-1], 1)
img = np.ones(cam_res, dtype = np.uint8)*255

ft = e_ts[0]
for i in range(0,len(e_ts)):
    #clear the frame at intervals
    if e_ts[i] >= ft:
        cv2.imshow('Visualisation', img)
        cv2.waitKey(np.int64(period*1000))
        img = np.ones(cam_res, dtype = np.uint8)*255
        ft = ft + period
    img[e_vs[i],e_us[i]] = 0
    
cv2.destroyAllWindows()

 # %% ---------------------------------
#DISPLAY ALL MARKER NAMES
for name in c3d_data.point_labels:
    print(name)

 # %% ---------------------------------
#PLOT 3D DATA

marker_names = {'*118', 'CLAV'}

for marker_name in marker_names:
    marker_points = helpers.marker_p(c3d_data.point_labels, points_3d.values(), marker_name)

    if marker_name == '*118':
        print("Initial Position", marker_points[0,0:3])
    #else:
    plt.plot(marker_t, marker_points[:, 0:3])
    plt.title(marker_name)
    plt.legend({'X', 'Y', 'Z'})
    plt.show()




# %%  ---------------------------------
# PROJECT 3D DATA

# create translation matrix
campos = helpers.marker_p(c3d_data.point_labels, points_3d.values(), '*118')

T = helpers.makeT([0, 8, -10], campos[0,0:3])
delay = 0.2
print(T)
print(delay)

#dict of marker names
marker_names = {'CLAV', 'RANK', 'LANK', 'LWRA', 'RWRA', 'LFHD', 'RFHD'}
#marker_names = {'CLAV'}

# extract points and convert them given translation matrix
image_points = {}
for mark_name in marker_names:
    ps = helpers.marker_p(c3d_data.point_labels, points_3d.values(), mark_name)
    # plt.plot(marker_t, ps[:, 0:3])
    # plt.title(mark_name + 'Raw')
    # plt.legend(['X', 'Y', 'Z'])
    # plt.show()

    ps = (T @ ps.transpose()).transpose()
    for p in ps:
        p /= p[-1]

    # plt.plot(marker_t, ps[:, 0:3])
    # plt.title(mark_name + 'RotTrans')
    # plt.legend(['X', 'Y', 'Z'])
    # plt.show()

    #only take positive points
    #ps = ps[ps[:, 2] > 0]

    #project onto the image plane given intrinsic parameters

    P_id = np.zeros((3, 4))
    #P_id[:, :3] = np.eye(3)
    P_id[0, 1] = -1
    P_id[1, 2] = -1
    P_id[2, 3] = 1
    image_points[mark_name] = (K @ P_id @ ps.transpose()).transpose()
    for p in image_points[mark_name]:
        p /= (1000*p[-1])

    plt.plot(marker_t, image_points[mark_name][:, 0:2])
    plt.legend(['U', 'V'])
    plt.plot([marker_t[0], marker_t[-1]], [cam_res[0], cam_res[0]], 'tab:orange', linestyle=':')
    plt.plot([marker_t[0], marker_t[-1]], [cam_res[1], cam_res[1]], 'b:')
    plt.plot([marker_t[0], marker_t[-1]], [0, 0], 'k:')
    plt.title(mark_name)
    plt.show()


#display on image
i_markers = 0
i_events = 0
tic_markers = marker_t[0] + period
tic_events = e_ts[0] + delay + period
img = np.ones(cam_res, dtype = np.uint8)*255

while tic_markers < marker_t[-1] and tic_events < e_ts[-1]:
    while marker_t[i_markers] < tic_markers:
        for mark_name in marker_names:
            u = np.int64(image_points[mark_name][i_markers][0]/2)
            v = np.int64(image_points[mark_name][i_markers][1]/2)
            cv2.circle(img, [u, v], 3, 0, cv2.FILLED)
            cv2.putText(img, mark_name, [u, v], cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
        i_markers = i_markers + 1

    while e_ts[i_events] < tic_events:
        img[e_vs[i_events],e_us[i_events]] = 0
        i_events = i_events + 1

    cv2.imshow('Projected Points', img)
    cv2.waitKey(np.int64(period*1000))
    img = np.ones(cam_res, dtype = np.uint8)*255
    tic_markers = tic_markers + period
    tic_events = tic_events + period

# img = np.ones(cam_res, dtype = np.uint8)*255
# ft = marker_t[0]
# for i in range(0,len(marker_t)):
#     #clear the frame at intervals
#     if marker_t[i] >= ft:
#         cv2.imshow('Projected Points', img)
#         cv2.waitKey(np.int64(period*1000))
#         img = np.ones(cam_res, dtype = np.uint8)*255
#         ft = ft + period
#     for mark_name in marker_names:
#         u = np.int64(image_points[mark_name][i][0]/2)
#         v = np.int64(image_points[mark_name][i][1]/2)
#         if u >= 0 and v >= 0 and u < cam_res[1] and v < cam_res[0]:
#             cv2.circle(img, [u, v], 3, 0, cv2.FILLED)
#             cv2.putText(img, mark_name, [u, v], cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
    
cv2.destroyAllWindows()


#plt image points over time
#plt.plot(image_points)
#plt.show()


 # %%
