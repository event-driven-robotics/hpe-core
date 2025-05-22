# okay first off lets load in all the data what do we need
# event data stream
# results file with transform and delay
# vicon file with marker positions 3D and camera marker positions 3D
# we also need the camera calibration
#
#%% SET-UP ARGUMENTS ---------------------------------
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from bimvee.importIitYarp import importIitYarp
import c3d
import helpers
import importlib
 
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

#%% ---------------------------------
#IMPORT EVENT DATA
v_data = importIitYarp(filePathOrName=args.events_path, zeroTimestamp=False)

#%% ---------------------------------
# IMPORT C3D DATA
c3d_data = c3d.Reader(open(args.c3d_path, 'rb'))
points_3d = {}
for i, points, analog in c3d_data.read_frames():
    points_3d[i] = points

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

ts = v_data['data']['left']['dvs']['ts']
us = v_data['data']['left']['dvs']['x']
vs = v_data['data']['left']['dvs']['y']

#range(ts[0]+1, ts[-1], 1)
img = np.ones(cam_res, dtype = np.uint8)*255

ft = ts[0]
for i in range(0,len(ts)):
    #clear the frame at intervals
    if ts[i] >= ft:
        cv2.imshow('Visualisation', img)
        cv2.waitKey(1)
        img = np.ones(cam_res, dtype = np.uint8)*255
        ft = ft + period
    img[vs[i],us[i]] = 0
    
cv2.destroyAllWindows()

 # %% ---------------------------------
#PLOT 3D DATA

#print(mark_name, i)
#print(c3d_data.point_labels)

#P1 might not only be 2 characters
#finding the index might be a faster way
mark_name = 'CLAV'

i = 0
for label in c3d_data.point_labels:
    k = label.find(':') + 1
    if(label[k:k+4] == mark_name):
        break
    i = i + 1

extract_p = lambda index: np.array([np.append(val[index][0:3], 1) for val in points_3d.values()])

print("Plot of joint:", mark_name, "at index", i)
plt.plot(extract_p(i))
plt.show()


# %%  ---------------------------------
# PROJECT 3D DATA
importlib.reload(helpers)

# create translation matrix
T = helpers.makeT(10, 10, 10)

# extract points and convert them given translation matrix
ps = extract_p(9) 
ps = (T @ ps.transpose()).transpose()
for p in ps:
    p /= p[-1]

#only take positive points
ps = ps[ps[:, 2] > 0]

#project onto the image plane given intrinsic parameters
P_id = np.zeros((3, 4))
P_id[:, :3] = np.eye(3)
image_points = (K @ P_id @ ps.transpose()).transpose()
for p in image_points:
    p /= p[-1]

#plt image points over time
plt.plot(image_points)
plt.show()


# %%
