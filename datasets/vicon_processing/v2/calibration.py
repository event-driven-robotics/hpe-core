# okay first off lets load in all the data what do we need
# event data stream
# results file with transform and delay
# vicon file with marker positions 3D and camera marker positions 3D
# we also need the camera calibration
#

# 
# import sys; sys.path.append('/home/aglover-iit.local/code/hpe-core/datasets/vicon_processing/v2')

#%% SET-UP ARGUMENTS ---------------------------------
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import c3d
import helpers
import importlib
import math
import yaml
import matplotlib.pyplot as plt

from helpers import ViconHelper
from helpers import DvsLabeler
from scipy.spatial.transform import Rotation as R

# import bimvee 2.0
sys.path.append('/usr/local/lib/bimvee')
from bimvee.importIitYarp import importIitYarp

sys.path.append('/home/cappe/hpe/hpe-core/datasets/vicon_processing/v2')

 
sys.argv = ['Interactive-1.ipynb', 
            '--events_path', '/home/cappe/hpe/move-iit-hpe-subset1/calibration/calib_test2/right/',
            '--c3d_path', '/home/cappe/hpe/move-iit-hpe-subset1/calibration/calib_test2_vicon/test11_calib01.c3d',
            '--calib_path', '/home/cappe/hpe/move-iit-hpe-subset1/calibration/calib_right.txt']

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

importlib.reload(helpers)

# marker_names = {'*118', 'CLAV', 'STRN'}
# marker_names = {'CLAV', 'STRN', 'RANK', 'LANK', 'LWRA', 'RWRA', 'LFHD', 'RFHD', 'LKNE', 'RKNE', 'LHIP', 'RHIP', 'RSHO', 'LSHO'}
#marker_names = {'*118', 'CLAV', 'STRN', 'RANK', 'LANK', 'LWRA', 'RWRA', 'LFHD', 'RFHD'}

marker_names = ['stereoatis:cam_right', 'stereoatis:cam_back', 'stereoatis:cam_left',
                'box:top_right_Origin', 'box:bottom_right', 'box:bottom_left', 'box:top_left']

for marker_name in marker_names:
    marker_points = helpers.marker_p(c3d_data.point_labels, points_3d.values(), marker_name)

    #if marker_name == '*118':
    #    print("Initial Position", marker_points[0,0:3])
    plt.plot(marker_t, marker_points[:, 0:3])
    plt.title(marker_name)
    plt.legend(['X', 'Y', 'Z'])
    plt.show()


# %% ---------------------------------
# LABEL SEQUENCE

# play video one temporal window at a time
# skip event frames with space bar, save labeled frame with 's', backspace to remove the last label
# when clicking on the screen, a list of markers is shown and you can select them

# automatically save it in a known path
labels_path = os.path.join(os.path.dirname(args.c3d_path), "labels.yml")
#print(f"labels path: {labels_path}")

time_tags, event_indices = helpers.calc_indices(e_ts, period)
e_tags = e_ts[event_indices]

img = np.ones(cam_res, dtype = np.uint8)*255

labeler = DvsLabeler(img_shape=(720, 1280, 3), subject=args.subject)
labeler.label_data(e_ts, e_us, e_vs, event_indices, time_tags, period)
labeler.save_labeled_points(labels_path)

print("Saved labeled points at:", labels_path)
    
cv2.destroyAllWindows()

# save yaml file with the labeled points, so to use them to calculate the transformation matrix with PNP
# TODO: save transformation matrix in a file with relative path

# %% ---------------------------------
# LOAD LABELS AND CALCULATE TRANSFORMATION MATRIX

time_tags, event_indices = helpers.calc_indices(e_ts, period)
e_tags = e_ts[event_indices]
#print("Event Tags: ", e_tags)
#print("Markers Tags: ", marker_t)
delay = e_tags[-1] - marker_t[-1]   # get time difference so to generate ids for the frames
# delay = -0.2 # ? -> check mainly for last cell that seems off

print("Delay: ", delay)

labels_path = os.path.join(os.path.dirname(args.c3d_path), "labels.yml")

labeled_points = helpers.read_points_labels(labels_path) # read labeled points from the yaml file
#print(labeled_points)
#labels_points = labeled_points['points']
#print("Labels Points: ", labels_points)
labels_time = labeled_points['times']
#print("Labels Time: ", labels_time)

vicon_helper = ViconHelper(marker_t, points_3d, delay, c3d_data.frame_count, c3d_data.point_rate, c3d_data.point_labels, None)

# extract the frames_id from the labeled points
frames_id = vicon_helper.get_frame_time(labeled_points['times'])
# print(f"the frames id are: {(frames_id)}")
# print(f"the frames timestamps are: {labeled_points['times']}")

# match the labeled points to the vicon points times
vicon_points = vicon_helper.get_vicon_points_interpolated(labeled_points)
# print(f"vicon points: {vicon_points}")

#print("Vicon marker names:", [l for l in helper.point_labels])
# print("First few label dicts from YAML:")
# for i, d in enumerate(labeled_points['points']):
#     print(f"Frame {i}: {d}")
    

# Print DVS label time, matched Vicon frame index, and Vicon frame time
# for dvs_time, vicon_idx in zip(labeled_points['times'], vicon_points['frame_ids']):
#     vicon_time = vicon_helper.frame_times[vicon_idx]
#     print(f"DVS time: {dvs_time:.6f}  |  Vicon frame idx: {vicon_idx}  |  Vicon time: {vicon_time:.6f}")
    
# print("Event Tags: ", e_tags)
# print("Markers Tags: ", marker_t)
# print("Labeled Tags: ", labels_time)

campos = helpers.marker_p(c3d_data.point_labels, points_3d.values(), 'stereoatis:cam_right')

world_points = []
image_points = []

for dvs_frame, vicon_frame in zip(labeled_points['points'], vicon_points['points']):
    labels = dvs_frame.keys()

    for l in labels:
        try:
            w_p = vicon_frame[l]
            i_p = [
                dvs_frame[l]['x'],
                dvs_frame[l]['y'] #,
                # 1.0
            ]

            world_points.append(w_p)
            image_points.append(i_p)
        except Exception as e:
            print("The stored image labels probably don't match with the vicon labels used.")
            print(e)

world_points = np.array(world_points, dtype=np.float64)
image_points = np.array(image_points, dtype=np.float64)
image_points = image_points[:, :2]

print("world_points shape:", world_points.shape)
print("image_points shape:", image_points.shape)

# Solve PnP, with 3D points from vicon and 2D points from labels
success, rvec, tvec = cv2.solvePnP(world_points, image_points, K, D)

# Convert rotation vector to rotation matrix
R_mat, _ = cv2.Rodrigues(rvec)

# Convert rotation matrix to Euler angles (degrees)
rot = R.from_matrix(R_mat)
angles = rot.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]

print("Rotation angles (degrees):", angles)
print("Translation vector:", campos[0, 0:3])

# Convert rotation matrix to Euler angles (degrees)
# rot = R.from_matrix(R_mat)
# angles = rot.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]

# print("Rotation angles (degrees):", angles)
# print("Translation vector:", campos[0, 0:3])

T = np.eye(4)
T[:3, :3] = R_mat
T[:3, 3] = tvec[:, 0]
print("Transformation Matrix T:\n", T)

# TODO: add back the initial guess to estimate the transformation matrix for the camera
# so that then we can introduce the method to match everything simply by clicking the two markers

# %% ---------------------------------
# PROJECT 3D DATA ONTO IMAGE PLANE

# TODO: check again delay calculation, as it seems to be off
 
# T = helpers.makeT(angles, tvec[:, 0])    # create tranformation matrix from computed angles and known traslation
# print("Transformation Matrix T:\n", T)

projected_points = helpers.project_vicon_to_event_plane(
    marker_names, c3d_data, points_3d, marker_t, T, K, cam_res, delay, e_ts, e_us, e_vs, period, visualize=True)

# %% ---------------------------------
# TEST WITH OTHER SEQUENCES

# load event and vicon data
# use same T to project the 3D points onto the image plane

new_events_path = '/home/cappe/hpe/move-iit-hpe-subset1/P1/walk_s1/atis-s/'
new_c3d_path = '/home/cappe/hpe/move-iit-hpe-subset1/P1/walk_s1.c3d'

new_ev_data = importIitYarp(filePathOrName=new_events_path, zeroTimestamp=False)
new_e_ts = new_ev_data['data']['left']['dvs']['ts']
new_e_us = new_ev_data['data']['left']['dvs']['x']
new_e_vs = new_ev_data['data']['left']['dvs']['y']

new_c3d_data = c3d.Reader(open(new_c3d_path, 'rb'))
new_points_3d = {}
for i, points, analog in new_c3d_data.read_frames():
    new_points_3d[i] = points

new_marker_t = np.linspace(
    0.0, new_c3d_data.frame_count / new_c3d_data.point_rate,
    new_c3d_data.frame_count, endpoint=False
)

projected_points = helpers.project_vicon_to_event_plane(
    marker_names, new_c3d_data, new_points_3d, new_marker_t, T, K, cam_res, delay, new_e_ts, new_e_us, new_e_vs, period, visualize=True)
