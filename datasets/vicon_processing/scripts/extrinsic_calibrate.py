import numpy as np
import sys
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
sys.path.append('/home/aglover-iit.local/code/hpe-core/datasets/')
sys.path.append('/home/cappe/hpe/hpe-core/datasets')

sys.path.append('/usr/local/lib/bimvee')
from bimvee.importIitYarp import importIitYarp

from vicon_processing.src.projection import ProjectionHelper
from vicon_processing.src.data_helpers import DvsLabeler, DvsHelper, C3dHelper
from vicon_processing.src import vis_utils, utils

parser = argparse.ArgumentParser(
                    prog='Extrinsic Calibrate',
                    description='Calibrate Pose of Camera in WRF')
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

parser.add_argument('--output_path', help='path where to save the transformation that can be used as extrinsic calibration', required=True)
parser.add_argument('--no_camera_markers', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

def setup_projection(delay):

    # import the DVS data
    dvs_file_path = args.dvs_path
    dvs_helper = DvsHelper(dvs_file_path)
    # read the labeled 2d points
    dvs_helper.read_points_labels(args.annotated_points);
    # extract the time of the labeled points
    print(dvs_helper.labeled_points)
    labels_times = dvs_helper.labeled_points['times'][:]

    print(labels_times)

    # load c3d vicon data
    c3d_file_path = args.vicon_path
    c3d_helper = C3dHelper(c3d_file_path, delay=delay, camera_markers=not args.no_camera_markers, filter_camera_markers=True)
    print(f"Labels in c3d file{c3d_helper.reader.point_labels}")

    vicon_labeled_frames = c3d_helper.get_frame_time(labels_times)
    print(f"frame time for vicon: {c3d_helper.frame_times}")

    vicon_points = c3d_helper.get_vicon_points_interpolated(dvs_helper.labeled_points)
    print(f"vicon points: {vicon_points}")
    vicon_points_mark = c3d_helper.transform_points_to_marker_frame(vicon_points)
    # vicon_points_mark = vicon_points
    print(f"vicon points in marker frame: {vicon_points_mark}")
    
    print(vicon_labeled_frames)
    c3d_helper.markers_T
    print(f"times from dvs labels: {dvs_helper.labeled_points['times']}")
    print(f"times from vicon labels: {vicon_points_mark['times']}")
    dvs_helper.labeled_points['times']
    vicon_points_mark['times']
    proj_helper = ProjectionHelper(vicon_points_mark, dvs_helper.labeled_points)
    proj_helper.import_camera_calbration(args.intrinsic);

    return proj_helper

def error_calib(delay):
    if delay < 0:
        return np.inf
    
    proj_helper = setup_projection(delay)
    # find transform
    T = proj_helper.find_R_t_opencv()

    error = proj_helper.measure_error(T)

    # print(f"measured error for d:{delay} -> {error}")
    return error

res = scipy.optimize.minimize_scalar(error_calib, tol=1e-9, bounds=(0.15, 0.4), method='bounded')

print(res)
best_delay = res.x
print(f"Best delay: {best_delay}")

proj_helper = setup_projection(best_delay)
best_T = proj_helper.find_R_t_opencv()

print(f"\nEstimated Transform: \n{best_T}\n")

np.save(args.output_path, best_T)
