#%%
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import c3d
import importlib

# Import helpers
sys.path.append('/app/hpe-core/datasets/vicon_processing/v2')
import helpers

# Import bimvee
sys.path.append('./submodules/bimvee')
from bimvee.importIitYarp import importIitYarp


#%%
# Load parameters
sys.argv = ['Interactive-1.ipynb', 
            '--events_folder', '/data/EventLS/calib_test2/Events/right',
            '--vicon_c3d_file', '/data/EventLS/calib_test2/VICON/test11_calib01.c3d',
            '--calib_path', '/data/EventLS/Calibration/calib_right.txt',
            '--label_tag_file', '../scripts/config/labels_tags_vicon.yml',]

parser = argparse.ArgumentParser(description="""Calibrate event camera with Vicon data.
Example usage:
|| uv run evcam_vicon_calib.py -e /data/EventLS/calib_test2/Events/right
-v /data/EventLS/calib_test2/VICON/test11_calib01.c3d
-c /data/EventLS/Calibration/calib_right.txt""")
parser.add_argument('--events_folder', type=str, required=True, help='Path to the event camera folder')
parser.add_argument('--vicon_c3d_file', type=str, required=True, help='Path to the Vicon C3D file')
parser.add_argument('--calib_path', type=str, required=True, help='Path to the calibration file')
parser.add_argument('--label_tag_file', type=str, default=None, help='Path to the label tag file')
parser.add_argument('--fps', default=25.0, help='Frames per second for the time window')

args = parser.parse_args()

vicon_c3d_file: str = args.vicon_c3d_file
events_folder: str = args.events_folder
calib_path: str = args.calib_path
period: float = 1.0 / args.fps
label_tag_file: str = args.label_tag_file

#%%
# Import event data
v_data: dict = importIitYarp(filePathOrName=events_folder, template=None)
e_ts: np.ndarray = v_data['data']['left']['dvs']['ts']
e_us: np.ndarray = v_data['data']['left']['dvs']['x']
e_vs: np.ndarray = v_data['data']['left']['dvs']['y']

#%%
# Import C3D data
c3d_data = c3d.Reader(open(vicon_c3d_file, 'rb'))
points_3d: dict = {}
for i, points, analog in c3d_data.read_frames():
    points_3d[i] = points

marker_t = np.linspace(0.0, c3d_data.frame_count / c3d_data.point_rate, 
                            c3d_data.frame_count, endpoint=False)
# %%
# Import Calibration data
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

# %%
# Visualize event data as time window

img = np.ones(cam_res, dtype = np.uint8)*255

ft = e_ts[0]
for i in range(0,len(e_ts)):
    #clear the frame at intervals
    if e_ts[i] >= ft:
        cv2.imshow('Visualisation', img)
        k = cv2.waitKey(np.int64(period*1000))
        if k == 27:  # ESC key to stop
            break
        img = np.ones(cam_res, dtype = np.uint8)*255
        ft = ft + period
    img[e_vs[i],e_us[i]] = 0
    
cv2.destroyAllWindows()
# %%
# label sequence
importlib.reload(helpers)
from helpers import DvsLabeler

labels_path = os.path.join(os.path.dirname(vicon_c3d_file), "labels.yml")

time_tags, event_indices = helpers.calc_indices(e_ts, period)

img = np.ones(cam_res, dtype = np.uint8)*255

labeler = DvsLabeler(img_shape=(720, 1280, 3), subject='box')
labeler.label_data(e_ts, e_us, e_vs, event_indices, time_tags, period, label_tag_file)
labeler.save_labeled_points(labels_path)

print("Saved labeled points at:", labels_path)
    
cv2.destroyAllWindows()
# %%
# Load labels and calculate transformation matrix
importlib.reload(helpers)
from helpers import ViconHelper
from scipy.spatial.transform import Rotation as R

e_tags = e_ts[event_indices]
delay = e_tags[-1] - marker_t[-1]   # get time difference so to generate ids for the frames
print("Delay: ", delay)

labels_path = os.path.join(os.path.dirname(vicon_c3d_file), "labels.yml")

labeled_points = helpers.read_points_labels(labels_path) # read labeled points from the yaml file
labels_time = labeled_points['times']

vicon_helper = ViconHelper(marker_t, points_3d, delay, c3d_data.frame_count, c3d_data.point_rate, c3d_data.point_labels, None) # if needed create the function to call instead of None

# extract the frames_id from the labeled points
frames_id = vicon_helper.get_frame_time(labeled_points['times'])
vicon_points = vicon_helper.get_vicon_points_interpolated(labeled_points)
print("First few label dicts from YAML:")
for i, d in enumerate(labeled_points['points']):
    print(f"Frame {i}: {d}")

# Solve PnP
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
# %%
# Project 3D world points to 2D image plane for confirmation
import matplotlib.pyplot as plt

def project_vicon_to_event_plane(
    marker_names, 
    c3d_data, 
    points_3d, 
    marker_t, 
    T, 
    cam_res, 
    delay, 
    e_ts, 
    e_us, 
    e_vs, 
    period
):
    # Project points from Vicon to event plane using the transformation matrix T

    projected_points = {}

    # For each marker, transform and project
    for mark_name in marker_names:
        ps = helpers.marker_p(c3d_data.point_labels, points_3d.values(), mark_name)
        # Homogenize if needed
        if ps.shape[1] == 3:
            ps = np.hstack([ps, np.ones((ps.shape[0], 1))])
        # Apply transformation
        ps_trans = (T @ ps.T).T
        ps_trans = ps_trans / ps_trans[:, [3]]  # Normalize homogeneous
        # Only keep 3D part
        ps_trans = ps_trans[:, :3]

        # Project to image plane
        img_pts, _ = cv2.projectPoints(ps_trans, np.zeros(3), np.zeros(3), K, None)
        img_pts = img_pts.reshape(-1, 2)
        projected_points[mark_name] = img_pts


    # Visualization
    img = np.ones(cam_res, dtype=np.uint8) * 255
    tic_markers = marker_t[0] + period
    tic_events = e_ts[0] + delay + period
    i_markers = 0
    i_events = 0

    while tic_markers < marker_t[-1] and tic_events < e_ts[-1]:
        while marker_t[i_markers] < tic_markers:
            for mark_name in marker_names:
                u = int(projected_points[mark_name][i_markers][0])
                v = int(projected_points[mark_name][i_markers][1])
                if 0 <= u < cam_res[1] and 0 <= v < cam_res[0]:
                    cv2.circle(img, (u, v), 3, 0, cv2.FILLED)
                    cv2.putText(img, mark_name, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
            i_markers += 1

        while e_ts[i_events] < tic_events:
            img[e_vs[i_events], e_us[i_events]] = 0
            i_events += 1

        cv2.imshow('Projected Points', img)
        c = cv2.waitKey(int(period * 1000))
        if c == ord('q'):
            cv2.destroyAllWindows()
            return projected_points
        img = np.ones(cam_res, dtype=np.uint8) * 255
        tic_markers += period
        tic_events += period

    cv2.destroyAllWindows()
    return projected_points

R_mat, _ = cv2.Rodrigues(rvec)
T_mat = np.zeros((4, 4))
T_mat[0:3, 0:3] = R_mat
T_mat[0:3, 3] = tvec[:, 0]
T_mat[3, 3] = 1.0
print("Transformation Matrix T:\n", T_mat)
marker_names = {'cam_right', 'cam_back', 'cam_left', 'top_right_Origin', 'bottom_right', 'bottom_left', 'top_left'}

projected_points = project_vicon_to_event_plane(
    marker_names=marker_names,
    c3d_data=c3d_data,
    points_3d=points_3d,
    marker_t=marker_t,
    T=T_mat,
    cam_res=cam_res,
    delay=delay,
    e_ts=e_ts,
    e_us=e_us,
    e_vs=e_vs,
    period=period
)

