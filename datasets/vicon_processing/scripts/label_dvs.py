import numpy as np
import sys
import matplotlib.pyplot as plt
import yaml
import cv2

sys.path.append('/home/schiavazza/code/hpe/')

from vicon_recordings.src.data_helpers import DvsLabeler, DvsHelper

# import the DVS data
dvs_file_path = '/home/schiavazza/data/hpe/vicon_recordings/stefano/1'
dvs_helper = DvsHelper(dvs_file_path)
dvs_helper.read_events()

# define the point labels to use
labels = [
    'P000:RASI',
    'P000:LASI',
    'P000:STRN',
    'P000:CLAV',
    'P000:RFHD',
    'P000:LFHD',
    # 'LSHO',
    'P000:RSHO',
    'P000:RELB',
    'P000:RUPA',
    'P000:LELB',
    'P000:LFRM',
    'P000:RKNE',
    'P000:RTIB',
]
frame_times = [4.0, 8.0, 10.0]

labeler = DvsLabeler(dvs_helper.events, (480, 640, 3))
out = labeler.label_data(frame_times, labels, duration=0.005)

# save labels
labeler.save_labeled_points('../data/points.yml')