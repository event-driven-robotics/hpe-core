import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from tqdm import tqdm
import argparse
import yaml

sys.path.append('/home/schiavazza/code/hpe/hpe-core/datasets/')
sys.path.append('/local_code/hpe-core/datasets/')
sys.path.append('/home/iit.local/schiavazza/local_code/hpe-core/datasets/')

from vicon_processing.src.projection import ProjectionHelper
from vicon_processing.src.data_helpers import DvsLabeler, DvsHelper, C3dHelper
from vicon_processing.src import vis_utils, utils


parser = argparse.ArgumentParser(
                    prog='Export cdf',
                    description='export the pose data in cdf format')
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
parser.add_argument('--output_path', help='path to output cdf', required=True)
parser.add_argument('--all_points', action=argparse.BooleanOptionalAction)
parser.add_argument('--camera_resolution', default=(1280, 720), nargs='+', type=int)
parser.add_argument('--vicon_delay', default=0.0, type=float)
parser.add_argument('--no_camera_markers', action=argparse.BooleanOptionalAction)
parser.add_argument('--subject',
                    required=True,
                    help="E.g. P11, P10 etc...")
parser.add_argument('--no_camera_filter',
                    action=argparse.BooleanOptionalAction)

args = parser.parse_args()


# import the DVS data
dvs_file_path = args.dvs_path
dvs_helper = DvsHelper(dvs_file_path)


# load c3d vicon data
c3d_file_path = args.vicon_path
c3d_helper = C3dHelper(c3d_file_path, 
                       delay=args.vicon_delay, 
                       camera_markers=not args.no_camera_markers, 
                       filter_camera_markers=not args.no_camera_filter)

with open(args.labels, "r") as stream:
    try:
        labels = yaml.safe_load(stream)
        labels = [f"{args.subject}:{l}" for l in labels]
    except yaml.YAMLError as exc:
        print(exc)

if args.all_points:
    labels = [l.strip() for l in c3d_helper.reader.point_labels]

print(f"Loaded labels: {labels}")

# import transformation
T = np.load(args.extrinsic)

print(f"using extrinsics: {T}")

proj_helper = ProjectionHelper()
proj_helper.import_camera_calbration(args.intrinsic);

# clean up the labels in case there are some white-spaces
labels = [l.strip() for l in labels]

def get_projected_points(frame_id):
    T_markers = c3d_helper.marker_T_at_frame_vector(frame_id)
    points_dict = c3d_helper.get_points_dict(frame_id)
    filtered_points = c3d_helper.filter_dict_labels(points_dict, labels)

    v_points = c3d_helper.points_dict_to_array(filtered_points)
    v_points = v_points[:, :4]
    v_points[:, -1] = 1
    
    projected_points = proj_helper.project_to_frame(
        proj_helper.transform_points(
            v_points, T @ T_markers
            )
        )
    
    return projected_points

resolution = args.camera_resolution

first_nonzero = np.searchsorted(c3d_helper.frame_times, 0.0)

os.makedirs(args.output_path, exist_ok=True)

data_file = os.path.join(args.output_path, "data.log")
f = open(data_file, "a")

for i in tqdm(range(first_nonzero, int(c3d_helper.reader.frame_count), 1)):
    projected_points = get_projected_points(i)[:, :2];
    
    f.write(f"{i - first_nonzero} {c3d_helper.frame_times[i]:.6f} SKLT ({' '.join(str(j) for j in projected_points.flatten().astype(int))})\n")

f.close()

info_file = os.path.join(args.output_path, "info.log")
f = open(info_file, "a")
f.write("Type: Bottle;\n[0.0] /file/ch0GTskeleton:o [connected]")
f.close()
    

print('Done')