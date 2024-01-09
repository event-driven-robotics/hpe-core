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
                    help='where to write the generated frames',
                    required=True)
parser.add_argument('--n_frames',
                    help='Number of frames to generate, the frames will be taken between start_time and end_time with linspace', 
                    default = 100, type=int)
parser.add_argument('--start_time',
                    help='Start time from which to take the frames, default: 2.0', 
                    default = 0.0, type=float)
parser.add_argument('--end_time',
                    help='End time for taking the frames, default: 15.0', 
                    default = 15.0, type=float)
parser.add_argument('--time_window',
                    help='Size of the time window to use for accumulating events', 
                    default = 0.02, type=float)
args = parser.parse_args()

frame_times = np.linspace(args.start_time, args.end_time, args.n_frames)

# import the DVS data
dvs_file_path = args.dvs_path
dvs_helper = DvsHelper(dvs_file_path)
dvs_helper.read_events()

labeler = DvsLabeler((720, 1280, 3), dvs_helper.events)
frames_folder = labeler.generate_frames(frame_times, args.output_path, duration=args.time_window)

print(f"created frames and saved them in {frames_folder} \n the frames times are saved in the .txt file")