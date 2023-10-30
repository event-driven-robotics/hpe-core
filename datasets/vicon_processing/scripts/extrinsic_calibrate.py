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

sys.path.append('/home/schiavazza/code/hpe/hpe-core/datasets/')

from vicon_processing.src.projection import ProjectionHelper
from vicon_processing.src.data_helpers import DvsLabeler, DvsHelper, C3dHelper
from vicon_processing.src import vis_utils, utils

parser = argparse.ArgumentParser(
                    prog='Generate video',
                    description='generate video from vicon data and dvs data, require a calibration transform')
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
parser.add_argument('--vicon_delay', default=0.0, type=float)


args = parser.parse_args()

# import the DVS data
dvs_file_path = args.dvs_path
dvs_helper = DvsHelper(dvs_file_path)
# read the labeled 2d points
dvs_helper.read_points_labels(args.annotated_points);
# extract the time of the labeled points
print(dvs_helper.labeled_points)
labels_times = dvs_helper.labeled_points['times']

print(labels_times)
# labels_times = [0.0]
labels = list(dvs_helper.labeled_points['points'][0].keys())
print(labels)


# load c3d vicon data
c3d_file_path = args.vicon_path
c3d_helper = C3dHelper(c3d_file_path, args.vicon_delay)
print(c3d_helper.reader.point_labels)
c3d_helper.reader.frame_count

vicon_labeled_frames = c3d_helper.get_frame_time(labels_times)
print(c3d_helper.frame_times)
vicon_points = c3d_helper.get_vicon_points(vicon_labeled_frames, labels)
vicon_points_mark = c3d_helper.transform_points_to_marker_frame(vicon_points)
# vicon_points_mark = vicon_points
print(vicon_labeled_frames)
c3d_helper.markers_T
print(f"times from dvs labels: {dvs_helper.labeled_points['times']}")
print(f"times from vicon labels: {vicon_points_mark['times']}")
dvs_helper.labeled_points['times']
vicon_points_mark['times']
proj_helper = ProjectionHelper(vicon_points_mark, dvs_helper.labeled_points)
proj_helper.import_camera_calbration(args.intrinsic);
proj_helper.image_points

# find transform
T = proj_helper.find_R_t()

np.save(args.output_path, T)
