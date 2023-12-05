import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
import argparse
import yaml

sys.path.append('/home/schiavazza/code/hpe/hpe-core/datasets/')
sys.path.append('/local_code/hpe-core/datasets/')

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
parser.add_argument('--labels', 
                    default='./config/labels.yml', 
                    help='file that defines the points labels that should be projected')
parser.add_argument('--intrinsic',
                    default='./config/temp_calib.txt', 
                    help='intrinsic calibration for the camera')
parser.add_argument('--extrinsic', default='./config/extrinsic_test.npy')
parser.add_argument('--frames_folder', help='path to frames folder', required=True)
parser.add_argument('--all_points', default=False)
parser.add_argument('--camera_resolution', default=(640, 480), nargs='+', type=int)
parser.add_argument('--vicon_delay', default=0.0, type=float)
parser.add_argument('--no_camera_markers', action=argparse.BooleanOptionalAction)

args = parser.parse_args()


# import the DVS data
dvs_file_path = args.dvs_path
dvs_helper = DvsHelper(dvs_file_path)


# load c3d vicon data
c3d_file_path = args.vicon_path
c3d_helper = C3dHelper(c3d_file_path, delay=args.vicon_delay, camera_markers=not args.no_camera_markers)

with open(args.labels, "r") as stream:
    try:
        labels = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if args.all_points:
    labels = [l.strip() for l in c3d_helper.reader.point_labels]

print(f"Loaded labels: {labels}")

# import transformation
T = np.load(args.extrinsic)

print(f"using extrinsics: {T}")

c3d_helper.find_start_time()

proj_helper = ProjectionHelper()
proj_helper.import_camera_calbration(args.intrinsic);

# clean up the labels in case there are some white-spaces
labels = [l.strip() for l in labels]

def get_projected_points(t):

    vicon_labeled_frames = c3d_helper.get_frame_time([t])
    T_markers = c3d_helper.marker_T_at_frame_vector(vicon_labeled_frames[0], t)
    vicon_points = c3d_helper.get_vicon_points_interpolated(vicon_labeled_frames, labels, [t])['points'][0]
    # vicon_points = c3d_helper.get_points_dict(vicon_labeled_frames[0])
    filtered_points = c3d_helper.filter_dict_labels(vicon_points, labels)

    v_points = c3d_helper.points_dict_to_array(filtered_points)
    v_points = v_points[:, :4]
    v_points[:, -1] = 1
    
    projected_points = proj_helper.project_to_frame(
        proj_helper.transform_points(
            v_points, T @ T_markers
            )
        )
    
    return projected_points

with open(os.path.join(args.frames_folder, "times.yml")) as f:
    times = yaml.load(f, Loader=yaml.Loader)

for t, filenameext in times:
    file_name, ext = os.path.splitext(filenameext)
    if ext != ".png":
        continue

    labeled_folder = os.path.join(args.frames_folder, "labeled")
    if not os.path.exists(os.path.join(labeled_folder, filenameext)):
        continue
    
    dvs_frame = cv2.imread(os.path.join(labeled_folder, filenameext))
    projected_points = get_projected_points(t)

    dvs_frame = vis_utils.plot_2d_points(dvs_frame, projected_points, color=(0, 0, 255), size=4)

    debug_folder = os.path.join(args.frames_folder, "debug")
    print
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    save_path = os.path.join(debug_folder, filenameext)
    try:
        cv2.imwrite(save_path, dvs_frame)
    except Exception as e:
        print(e)



print('Done')