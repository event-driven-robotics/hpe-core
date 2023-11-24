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
                    prog='Generate frames',
                    description='generate frames from dvs data')
parser.add_argument('--dvs_path', 
                    required=True, 
                    help='path to the yarp folder containing the dvs recording')

parser.add_argument('--output_path', 
                    help='where to write the generated frames. It will generate a yaml file', 
                    required=True)

args = parser.parse_args()

# import the DVS data
dvs_file_path = args.dvs_path
dvs_helper = DvsHelper(dvs_file_path)
dvs_helper.read_events()

# TODO make the times a parameter that can be set
frame_times = np.linspace(1, 15, 20)
labeler = DvsLabeler((720, 1280, 3), dvs_helper.events)
frames_folder = labeler.generate_frames(frame_times, args.output_path, duration=0.05)

print(f"created frames and saved them in {frames_folder} \n the frames times are saved in the .txt file")