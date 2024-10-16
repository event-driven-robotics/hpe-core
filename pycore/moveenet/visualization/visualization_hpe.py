import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import math
from tqdm import tqdm
import cv2
from pathlib import Path
from scipy import interpolate
from tabulate import tabulate
from typing import Optional

from datasets.utils import constants as ds_constants, parsing as ds_parsing
from evaluation.utils import metrics as metrics_utils, plots as plots_utils
from evaluation.utils.plots import plot_poses
from visualization import add_skeleton

def viz_prediction_all_joints(skeletons_gt, skeletons_predictions):
     res = [480, 640]
     image = np.zeros(res, np.uint8)
     thickness = 2
     count = 0
     image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2BGR)
     image = add_skeleton(image, skeletons_predictions[0], (255, 0, 0), lines=True, normalised=False)
     image = cv2.resize(image, (res[1], res[0]))
     
     pass

def main(args):
    plt.close('all')
    output_folder_path = Path('/home/cpham-iit.local/data/h36m/videos/')

    # predictions_folder = Path(args.predictions_path)
    predictions_path = Path('/home/cpham-iit.local/data/eval/cam2_S1_Directions')

    predictions_file_path = list(predictions_path.glob('**/*.csv'))
    print(predictions_file_path)
    # if len(predictions_file_path) == 0:
    #     print('\x1b[1;33;20m' + "Skipping " + " as no results exist in" + str(
    #         predictions_path) + '\x1b[0m')
    #     continue
    res = [480, 640]

    predictions_file_path.sort(reverse=True)
    algorithm_names = []
    skeletons_predictions = []
    timestamps = []
    latency = []

    #parse prediction
    for pred_path in predictions_file_path:
        print('pred path',pred_path)
        algo_name = pred_path.stem 
        if algo_name in args.exclude:
            continue
        try:
            predictions_old = np.loadtxt(str(pred_path.resolve()),dtype= float)
        except ValueError:
            with open(str(pred_path.resolve())) as f:
                    content = f.readlines()
            for l, line in enumerate(content):
                 predictions_old[l,:] = np.asarray(line.split(','))

        predictions_old = predictions_old[predictions_old[:, 0].argsort()]

        idx = np.where(np.logical_and(predictions_old[:, 0] > 5, predictions_old[:, 0] < 15))
        predictions = predictions_old[idx[0], :]
        print('predictions: ', predictions.shape)
        ts_pred = predictions[:,0]
        print('time stamps: ', ts_pred.shape)
        timestamps.append(ts_pred)
        skeletons_pred = predictions[:, 2:].reshape(len(predictions),
                                                        len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)
        algorithm_names.append(algo_name)
        skeletons_predictions.append(skeletons_pred)
        print('skt pred',len(skeletons_predictions))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-p', '--predictions_path',
                        help='Path to the predictions folder containing subfolders with results in .csv format.',
                        required=False)
    parser.add_argument('-o', '--output_folder', help='Path to the folder where evaluation results will be saved',
                        required=False)
    parser.add_argument('-lat', help='flag specifying that the latency must be computed', dest='lat',
                        action='store_true')
    parser.set_defaults(lat=False)
    parser.add_argument('-e', '--exclude', action='append', default=[],
                        help='Exclude specific algorithms from results. Add a new -e for each algo.', required=False)
    args, unknown = parser.parse_known_args()
    if (unknown):
        print('\x1b[1;31;20m' + 'Unknown argument/s: ' + ' '.join(unknown) + '\x1b[0m')

    main(args)