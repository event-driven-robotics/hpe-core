import numpy as np
import sys
import matplotlib.pyplot as plt
import yaml
import cv2
import argparse

sys.path.append('/home/schiavazza/code/hpe/hpe-core/datasets/')
sys.path.append('/local_code/hpe-core/datasets/')

from vicon_processing.src.data_helpers import DvsLabeler, DvsHelper

parser = argparse.ArgumentParser(
                    prog='Generate video',
                    description='generate video from vicon data and dvs data, require a calibration transform')
parser.add_argument('--dvs_path', 
                    required=True, 
                    help='path to the yarp folder containing the dvs recording')
parser.add_argument('--calib_labels', 
                    default='./config/calib_labels.yml', 
                    help='file that defines the points labels used for the labeling and calibration')
parser.add_argument('--frames_path', 
                    required=True, 
                    help='path to the folder containing the generated dvs frames')

parser.add_argument('--output_path', 
                    help='where to write the labeled points. It will generate a yaml file', 
                    required=True)

args = parser.parse_args()

# import the DVS data
dvs_file_path = args.dvs_path
dvs_helper = DvsHelper(dvs_file_path)

# define the point labels to use
with open(args.calib_labels, "r") as stream:
    try:
        labels = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# TODO make the times a parameter that can be set
labeler = DvsLabeler((720, 1280, 3))
out = labeler.label_data(args.frames_path, labels)
labeler.save_labeled_points(args.output_path)