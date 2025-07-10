import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation
import yaml
import cv2
from scipy.spatial.transform import Rotation
from matplotlib.patches import Rectangle
from tqdm import tqdm
import argparse
import scipy.optimize

sys.path.append('/home/schiavazza/code/hpe/hpe-core/datasets/')
sys.path.append('/local_code/hpe-core/datasets/')
sys.path.append('/home/cappe/hpe/hpe-core/datasets')


from vicon_processing.src.projection import ProjectionHelper
from vicon_processing.src.data_helpers import DvsLabeler, DvsHelper, C3dHelper
from vicon_processing.src import vis_utils, utils

parser = argparse.ArgumentParser(
                    prog='optimise delay',
                    description='find the optimial time the minimises the error')
parser.add_argument('--dvs_path', 
                    required=True, 
                    help='path to the yarp folder containing the dvs recording')
parser.add_argument('--vicon_path', 
                    required=True, 
                    help='path to the vicon data in .c3d file format')
parser.add_argument('--annotated_points', 
                    default='./config/points.yml', 
                    help='file that defines the points labels that should be projected')
parser.add_argument('--intrinsic',
                    default='./config/temp_calib.txt', 
                    help='intrinsic calibration for the camera')
parser.add_argument('--extrinsic', default=None, help="the extrinsic transformation to use")
parser.add_argument('--output', required=True, help="txt file where to write the optmial delay found")

parser.add_argument('--no_camera_markers', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

# import the DVS data
dvs_file_path = args.dvs_path
dvs_helper = DvsHelper(dvs_file_path)
# read the labeled 2d points
dvs_helper.read_points_labels(args.annotated_points);
# extract the time of the labeled points
print(dvs_helper.labeled_points)
labels_times = dvs_helper.labeled_points['times'][:]

print(labels_times)
labels = list(dvs_helper.labeled_points['points'][0].keys())
print(labels)

min_error = np.inf
best_delay = 0.2

def error_delay(delay):
    if delay < 0:
        return np.inf

    print(f"Trying delay: {delay}")
    c3d_file_path = args.vicon_path
    c3d_helper = C3dHelper(c3d_file_path, delay=delay, camera_markers=not args.no_camera_markers, filter_camera_markers=False)

    vicon_labeled_frames = c3d_helper.get_frame_time(labels_times)

    vicon_points = c3d_helper.get_vicon_points_interpolated(dvs_helper.labeled_points)
    vicon_points_mark = c3d_helper.transform_points_to_marker_frame(vicon_points)
    # vicon_points_mark = vicon_points
    dvs_helper.labeled_points['times']
    vicon_points_mark['times']
    proj_helper = ProjectionHelper(vicon_points_mark, dvs_helper.labeled_points)
    proj_helper.import_camera_calbration(args.intrinsic);
    proj_helper.image_points

    T = np.load(args.extrinsic)

    error = proj_helper.measure_error(T)
    # print(f"measured error for d:{delay} -> {error}")
    return error

res = scipy.optimize.minimize_scalar(error_delay, tol=1e-9, bounds=(0.15, 0.4), method='bounded')

print(res)
best_delay = res.x
print(f"Best delay: {best_delay}")
#np.savetxt(os.path.join(args.dvs_path, f"../{args.output}"), [best_delay], fmt="%.9f")
np.savetxt(args.output, [best_delay], fmt="%.9f")