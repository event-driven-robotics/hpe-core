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

from typing import Tuple, Optional


from helpers import ViconHelper
from helpers import DvsLabeler
from scipy.spatial.transform import Rotation as R

# import bimvee 2.0
sys.path.append('/home/cappe/hpe/hpe-core/datasets/vicon_processing/v2/submodules/bimvee')
from bimvee.importIitYarp import importIitYarp

sys.path.append('/home/cappe/hpe/hpe-core/datasets/vicon_processing/v2')

 
# sys.argv = ['Interactive-1.ipynb', 
#             '--events_path', '/home/cappe/hpe/move-iit-hpe-subset1/calibration/calib_test1/left/',
#             '--c3d_path', '/home/cappe/hpe/move-iit-hpe-subset1/calibration/calib_test1_vicon/test10_calib02.c3d',
#             '--calib_path', '/home/cappe/hpe/move-iit-hpe-subset1/calibration/calib_left.txt']

sys.argv = ['Interactive-1.ipynb', 
            '--events_path', '/home/cappe/hpe/move-iit-hpe-subset1/calibration/test6/left/',
            '--c3d_path', '/home/cappe/hpe/move-iit-hpe-subset1/calibration/test6_vicon/test6.c3d',
            '--calib_path', '/home/cappe/hpe/move-iit-hpe-subset1/calibration/calib_left.txt']


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
v_data = importIitYarp(filePathOrName=args.events_path, zeroTimestamp=False, stop_ratio=0.25)

# v_data = importIitYarp(filePathOrName=args.events_path, zeroTimestamp=False)
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

marker_names = ['stereoatis:cam_right', 'stereoatis:cam_back', 'stereoatis:cam_left', 'box1:top_1_origin',
    'box1:top_2', 'box1:top_3', 'box1:top_4', 'box1:bottom_1', 'box1:bottom_2', 'box1:bottom_3', 'box1:bottom_4']

for marker_name in marker_names:
    marker_points = helpers.marker_p(c3d_data.point_labels, points_3d.values(), marker_name)

    if marker_name == 'stereoatis:cam_right':
        print("Initial Position right", marker_points[0,0:3])
    elif marker_name == 'stereoatis:cam_back':
        print("Initial Position back", marker_points[0,0:3])
    elif marker_name == 'stereoatis:cam_left':
        print("Initial Position left", marker_points[0,0:3])
        
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
labels_path = os.path.join(os.path.dirname(args.c3d_path), "labels_left.yml")
#print(f"labels path: {labels_path}")

time_tags, event_indices = helpers.calc_indices(e_ts, period)
e_tags = e_ts[event_indices]

img = np.ones(cam_res, dtype = np.uint8)*255

labeler = DvsLabeler(img_shape=(720, 1280, 3))
labeler.label_data(e_ts, e_us, e_vs, event_indices, time_tags, period)
labeler.save_labeled_points(labels_path)

print("Saved labeled points at:", labels_path)
    
cv2.destroyAllWindows()

# save yaml file with the labeled points, so to use them to calculate the transformation matrix with PNP

# %% ---------------------------------
# LOAD LABELS AND CALCULATE TRANSFORMATION MATRIX

time_tags, event_indices = helpers.calc_indices(e_ts, period)
e_tags = e_ts[event_indices]
print("Event Tags: ", e_tags)
print("Markers Tags: ", marker_t)
delay = e_tags[-1] - marker_t[-1]   # get time difference so to generate ids for the frames
# delay = -0.2 # ? -> check mainly for last cell that seems off

# TODO: correct delay calculation, as it seems to be off

print("Delay: ", delay)

labels_path = os.path.join(os.path.dirname(args.c3d_path), "labels_left.yml")

labeled_points = helpers.read_points_labels(labels_path) # read labeled points from the yaml file
#print(labeled_points)
#labels_points = labeled_points['points']
#print("Labels Points: ", labels_points)
labels_time = labeled_points['times']
#print("Labels Time: ", labels_time)

vicon_helper = ViconHelper(marker_t, points_3d, delay, c3d_data.frame_count, c3d_data.point_rate, c3d_data.point_labels, True, True)

# extract the frames_id from the labeled points
frames_id = vicon_helper.get_frame_time(labeled_points['times'])
# print(f"the frames id are: {(frames_id)}")
# print(f"the frames timestamps are: {labeled_points['times']}")

# match the labeled points to the vicon points times
vicon_points = vicon_helper.get_vicon_points_interpolated(labeled_points)
# print(f"vicon points: {vicon_points}")

# TODO: check again delay calculation, as it seems to be off

#campos = helpers.marker_p(c3d_data.point_labels, points_3d.values(), 'stereoatis:cam_left')

world_points = []
image_points = []

for dvs_frame, vicon_frame in zip(labeled_points['points'], vicon_points['points']):
    labels = dvs_frame.keys()

    for l in labels:
        try:
            w_p = vicon_frame[l]
            i_p = [
                dvs_frame[l]['x'],
                dvs_frame[l]['y']
            ]

            world_points.append(w_p)
            image_points.append(i_p)
        except Exception as e:
            print("The stored image labels probably don't match with the vicon labels used.")
            print(e)

world_points = np.array(world_points, dtype=np.float64)
image_points = np.array(image_points, dtype=np.float64)
world_points = world_points[:, :3]
image_points = image_points[:, :2]

# Get transformation matrix from vicon to camera markers system vector
# T_world_to_system, camera_vector_vicon_points, timestamps = vicon_helper.world_to_camera_markers(vicon_points)

# Solve PnP, with 3D points from vicon and 2D points from labels
success, rvec, tvec = cv2.solvePnP(world_points, image_points, K, D)

# Convert rotation vector to rotation matrix
R_mat, _ = cv2.Rodrigues(rvec)

T_world_to_camera = np.eye(4)
T_world_to_camera[:3, :3] = R_mat
T_world_to_camera[:3, 3] = tvec[:, 0]
print("Transformation Matrix T_world_to_camera:\n", T_world_to_camera)

# TODO: save transformation matrix in a file with relative path

# TODO: add back the initial guess to estimate the transformation matrix for the camera
# so that then we can introduce the method to match everything simply by clicking the two markers

# # %% ---------------------------------
# # LOAD LABELS AND CALCULATE TRANSFORMATION MATRICES FOR STATIC INTERVALS, DEBUG KINDA

# # Build world_points, image_points, and points_time for static intervals
# world_points = []
# image_points = []
# points_time = []

# for dvs_frame, vicon_frame, t in zip(labeled_points['points'], vicon_points['points'], labels_time):
#     labels = dvs_frame.keys()
#     for l in labels:
#         try:
#             w_p = vicon_frame[l]
#             i_p = [
#                 dvs_frame[l]['x'],
#                 dvs_frame[l]['y']
#             ]
#             world_points.append(w_p)
#             image_points.append(i_p)
#             points_time.append(t)
#         except Exception as e:
#             print("The stored image labels probably don't match with the vicon labels used.")
#             print(e)

# world_points = np.array(world_points, dtype=np.float64)
# image_points = np.array(image_points, dtype=np.float64)
# points_time = np.array(points_time, dtype=np.float64)
# world_points = world_points[:, :3]
# image_points = image_points[:, :2]

# T_world_to_camera_list = []

# for start, end in static_intervals:
#     # Find indices of labeled points within this static interval
#     mask = (points_time >= start) & (points_time <= end)
#     if np.sum(mask) < 4:
#         # Not enough points for PnP, skip this interval
#         continue

#     world_pts_interval = world_points[mask]
#     image_pts_interval = image_points[mask]

#     # Run PnP for this interval
#     success, rvec, tvec = cv2.solvePnP(world_pts_interval, image_pts_interval, K, D)
#     if not success:
#         print(f"PnP failed for interval {start}-{end}")
#         continue

#     R_mat, _ = cv2.Rodrigues(rvec)
#     T = np.eye(4)
#     T[:3, :3] = R_mat
#     T[:3, 3] = tvec[:, 0]
#     T_world_to_camera_list.append(T)
#     print(f"Interval {start:.2f}-{end:.2f}: Transformation Matrix:\n{T}")

# print(f"Computed {len(T_world_to_camera_list)} static transformation matrices.")

# # TODO: save transformation matrices in a file with relative path if needed

# %% ---------------------------------
# INTERPOLATION

importlib.reload(helpers)
from helpers import ViconHelper
from typing import List

time_tags, event_indices = helpers.calc_indices(e_ts, period)
e_tags = e_ts[event_indices]
delay = e_tags[-1] - marker_t[-1]   # get time difference so to generate ids for the frames
print("Delay: ", delay)

labels_path = os.path.join(os.path.dirname(args.c3d_path), "labels_left.yml")

labeled_points = helpers.read_points_labels(labels_path) # read labeled points from the yaml file
labels_time = labeled_points['times']

vicon_helper = ViconHelper(marker_t, points_3d, delay, c3d_data.frame_count, c3d_data.point_rate, c3d_data.point_labels, True, True) # if needed create the function to call instead of None

# extract the frames_id from the labeled points
frames_id = vicon_helper.get_frame_time(labeled_points['times'])
vicon_points = vicon_helper.get_vicon_points_interpolated(labeled_points)
print("First few label dicts from YAML:")
for i, d in enumerate(labeled_points['points']):
    print(f"Frame {i}: {d}")

# Solve PnP
system_points = []
image_points = []
Ts_system_to_camera: List[np.ndarray] = []
Ts_world_to_system = vicon_helper.compute_camera_marker_transforms(c3d_data, points_3d)

for idx, (dvs_frame, vicon_frame) in enumerate(zip(labeled_points['points'], vicon_points['points'])):
    labels = dvs_frame.keys()

    for l in labels:
        try:
            w_p = vicon_frame[l]
            w_ph = np.append(w_p, 1.0)  # homogeneous coordinates
            i_p = [
                dvs_frame[l]['x'],
                dvs_frame[l]['y'] #,
                # 1.0
            ]
            frame_id = vicon_points['frame_ids'][idx]
            p_sys = Ts_world_to_system[frame_id] @ w_ph  # convert to system coordinates
            system_points.append(p_sys[:3])
            image_points.append(i_p)
        except Exception as e:
            print("The stored image labels probably don't match with the vicon labels used.")
            print(e)

system_points = np.array(system_points, dtype=np.float64)
image_points = np.array(image_points, dtype=np.float64)
image_points = image_points[:, :2]

# %% ---------------------------------
# INIITIALIZATION AND OPTIMIZATION OF THE TRANSFORMATION MATRIX?

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def reprojection_error(params, Ps, pc, K, dist):
    """
    params: 6次元 (rotation_vector [3], translation [3])
    Ps: (N, 3) 3D点（システム座標系）
    pc: (N, 2) 2D点（画像座標系）
    K: (3, 3) カメラ行列
    dist: (5,) 歪みパラメータ（OpenCV形式）
    """
    rvec = params[:3]
    tvec = params[3:6]

    # OpenCVのprojectPointsで再投影
    projected_points, _ = cv2.projectPoints(Ps, rvec, tvec, K, dist)
    projected_points = projected_points.reshape(-1, 2)

    return (projected_points - pc).ravel()  # フラット化して返す

def estimate_Tstoc(Ps, pc, K, dist, init_params=None):
    """
    Ps: (N, 3) 3D点（システム座標系）
    pc: (N, 2) 2D点（画像座標系）
    K: (3, 3) 内部パラメータ
    dist: (5,) 歪み
    return: (4, 4) システム→カメラ座標変換行列 Tstoc
    """

    # 初期値: ゼロ回転、ゼロ並進
    if init_params is None:
        # 初期値をゼロに設定
        init_params = np.zeros(6)

    # 最小化（LM法などを選択可）
    res = least_squares(
        reprojection_error,
        init_params,
        args=(Ps, pc, K, dist),
        method='lm'  # Levenberg-Marquardt
    )

    rvec_opt = res.x[:3]
    tvec_opt = res.x[3:6]
    R_opt, _ = cv2.Rodrigues(rvec_opt)

    # 同次変換行列を構成
    T = np.eye(4)
    T[:3, :3] = R_opt
    T[:3, 3] = tvec_opt

    return T, res



# how is init_T defined?
init_T = np.array([[ 5.54118388e-01, -6.46688971e-01, 5.24162298e-01, -1.25350531e+02], [-3.65516271e-01, -7.54740636e-01, -5.44760888e-01, -8.06780263e+00], [ 7.47897430e-01, 1.10272191e-01, -6.54591006e-01, -9.41323412e+01], [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# init_T = T_world_to_camera

# TODO: manual delay adjustment
# TODO: definition of init_T



# traslation from ruler mesures
# rotation manually adjusted

# tvec = [50, 115, 5] # ruler measured traslation vector in mm

r_vec = Rotation.from_matrix(init_T[:3, :3]).as_rotvec()
t_vec = init_T[:3, 3]
r_vec, t_vec, init_T
init_param = np.concatenate((r_vec, t_vec))
init_param

T_w2c_opt, res = estimate_Tstoc(system_points, image_points, K, D, init_param)
T_w2c_opt, res

# %% ---------------------------------
# PROJECT 3D DATA ONTO IMAGE PLANE, "STATIC"

importlib.reload(helpers)

# T_world_to_camera = np.array([[ 5.54118388e-01, -6.46688971e-01, 5.24162298e-01, -1.25350531e+02], [-3.65516271e-01, -7.54740636e-01, -5.44760888e-01, -8.06780263e+00], [ 7.47897430e-01, 1.10272191e-01, -6.54591006e-01, -9.41323412e+01], [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

projected_points = helpers.project_vicon_to_event_plane(
    marker_names, c3d_data, points_3d, marker_t, T_world_to_camera, K, cam_res, delay, e_ts, e_us, e_vs, period, visualize=True, D=D)

# %% ---------------------------------
# DYNAMICALLY PROJECT 3D DATA ONTO IMAGE PLANE

importlib.reload(helpers)

# TODO: define transformation from camera markers system to actual camera
# TODO: compute transformation matrices for each frame based on camera markers positions
# TODO: correct projection each frame based on the first transformation matrix
# TODO: use the transformation matrices to project the points dynamically so that it follows the movement correctlyù
# TODO: do this for all marker_T?

# iterate through all frames and compute the transformation matrix for each frame
#T_camera_markers, vicon_points_camera_vector = vicon_helper.world_to_camera_markers(vicon_points)  #?

import helpers

# TODO: differentiate between project and project_dynamic?

# Compute T_camera_markers for every marker_t

# Calculate the list of all transformation matrices from world to system for each vicon reading
T_world_to_system = vicon_helper.compute_camera_marker_transforms(c3d_data, points_3d)    # TODO: events not vicon?
# T_world_to_system = vicon_helper.compute_camera_marker_transforms_interpolated()

# TODO: manual optimization of the transformation matrix
# TODO: translation from camera to camera markers system

rot_angles = [0.0, -8.0, -8.0]  # rx, ry, rz in degrees
translation = [-50, 115, 5]    # x, y, z in mm

# Create rotation matrix from Euler angles
R_mat = Rotation.from_euler('xyz', rot_angles, degrees=True).as_matrix()

# Build the 4x4 transformation matrix
T_system_to_camera = np.eye(4)
T_system_to_camera[:3, :3] = R_mat
T_system_to_camera[:3, 3] = translation

# get first transformation matrix from world to system and use it to compute the transformation from camera system to camera
#T_system_to_camera = T_world_to_camera @ np.linalg.inv(T_world_to_system[0])

print("camera to camera markers transformation matrix:\n", T_system_to_camera)

projected_points = helpers.project_vicon_to_event_plane_dynamic_manually_adjust_rotation(
    marker_names, c3d_data, points_3d, marker_t, T_system_to_camera, T_world_to_system, K, cam_res, delay, e_ts, e_us, e_vs, period, visualize=True, D=D
)

# # %% ---------------------------------
# # DINAMICALLY PROJECT 3D DATA ONTO IMAGE PLANE, STATIC INTERVALS

# importlib.reload(helpers)

# T_world_to_system = vicon_helper.compute_camera_marker_transforms(c3d_data, points_3d)    # TODO: events not vicon?

# T_system_to_camera_list = []

# for i in range(len(T_world_to_camera_list)):
#     idx = np.argmin(np.abs(marker_t - static_starts[i]))
#     T_system_to_camera_list.append(T_world_to_camera_list[i] @ np.linalg.inv(T_world_to_system[idx]))

# print("camera to camera markers transformation matrix list:\n", T_system_to_camera_list)

# # Call the dynamic projection function, passing the per-frame transformation matrices
# projected_points = helpers.project_vicon_to_event_plane_dynamic(
#     marker_names, c3d_data, points_3d, marker_t, T_system_to_camera_list, T_world_to_system, K, cam_res, delay, e_ts, e_us, e_vs, period, visualize=True, D=D, static_starts=static_starts
# )

# %% ---------------------------------
# TEST WITH OTHER SEQUENCES

# load event and vicon data
# use same T to project the 3D points onto the image plane

# too many events, cannot load everything in memory
new_events_path = '/home/cappe/hpe/move-iit-hpe-subset1/calibration/test6/right/'
new_c3d_path = '/home/cappe/hpe/move-iit-hpe-subset1/calibration/test6_vicon/test6.c3d'

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
    marker_names, new_c3d_data, new_points_3d, new_marker_t, T, K, cam_res, delay, new_e_ts, new_e_us, new_e_vs, period, visualize=True, D=D)

# %% ---------------------------------
# GET TRANSFORMATION MATRIX FROM VICON TO CAMERA MARKERS SYSTEM

# TODO: define world_to_camera_markers ?? -> given known markers position (cam_right, left, etc)

# marker_T = 
# vicon_helper = ViconHelper(marker_t, points_3d, delay, c3d_data.frame_count, c3d_data.point_rate, c3d_data.point_labels, marker_T)
