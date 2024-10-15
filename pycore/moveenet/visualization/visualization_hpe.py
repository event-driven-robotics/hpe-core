import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import math
from tqdm import tqdm

from pathlib import Path
from scipy import interpolate
from tabulate import tabulate
from typing import Optional

from datasets.utils import constants as ds_constants, parsing as ds_parsing
from evaluation.utils import metrics as metrics_utils, plots as plots_utils
from evaluation.utils.plots import plot_poses
from visualization import add_skeleton

def viz_prediction_all_joints(output_folder_path, skeletons_gt, skeletons_predictions):
     
     pass

def main(args):
    plt.close('all')
    output_folder_path = Path(args.output_folder).reslove()
    output_folder_path.mkdir(parents=True, exist_ok=True)

    predictions_folder = Path(args.predictions_path)
    predictions_path = predictions_folder / sample

    predictions_file_path = list(predictions_path.glob('**/*.csv'))

    # if len(predictions_file_path) == 0:
    #     print('\x1b[1;33;20m' + "Skipping " + " as no results exist in" + str(
    #         predictions_path) + '\x1b[0m')
    #     continue

    predictions_file_path.sort(reverse=True)
    algorithm_names = []
    skeletons_predictions = []
    timestamps = []
    latency = []

    #parse prediction
    for pred_path in predictions_file_path:
        algo_name = pred_path.stem 
        if algo_name in args.exclude:
            continue
        try:
            predictions_old = np.load.txt(str(pred_path.resolve()),dtype= float)
        except ValueError:
            with open(str(pred_path.resolve())) as f:
                    content = f.readlines()
            for l, line in enumerate(content):
                 predictions_old[l,:] = np.asarray(line.split(','))

        predictions_old = predictions_old[predictions_old[:, 0].argsort()]

        idx = np.where(np.logical_and(predictions_old[:, 0] > 5, predictions_old[:, 0] < 15))
        predictions = predictions_old[idx[0], :]

        ts_pred = predictions[:,0]
        timestamps.append(ts_pred)
        skeletons_pred = predictions[:, 2:].reshape(len(predictions),
                                                        len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)
        algorithm_names.append(algo_name)
        skeletons_predictions.append(skeletons_pred)

        