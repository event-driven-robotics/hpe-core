
import argparse
import cv2
import numpy as np
import os

from pathlib import Path
from scipy import interpolate
from tabulate import tabulate

from datasets.utils import constants as ds_constants, parsing as ds_parsing
from evaluation.utils import metrics as metrics_utils, plots as plots_utils


# - import predictions (from csv?) into numpy matrix
# - compute
#   - pck (head or torso according to param)
#     - for a set of thresholds
#     - for each joint
#    - global
#   - plot pck auc
#   - compute oks
#     - for a set of thresholds
#     - for each joint
#    - global
# - sample frames randomly and plot joints

def main(args):

    metrics = ['pck', 'rmse']

    results = dict()
    results['datasets'] = dict()
    results['algos'] = dict()

    # - import gt (from yarp?) into numpy matrix (from all validation recordings)
    datasets_path = Path(args.datasets_path)
    yarp_file_paths = list(datasets_path.glob('**/data.log'))
    for yarp_path in yarp_file_paths:

        if 'skeleton' not in yarp_path.parent.name:
            continue

        data = ds_parsing.import_yarp_skeleton_data(yarp_path)

        # TODO: make it general to 2d/3d coordinates
        ts_pred = np.concatenate(([.0], data['ts']))

        # interpolate ground truth joints so that they can be compared with the high frequency predictions
        for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
            x_interpolation = interpolate.interp1d(ts_pred, np.concatenate(([data[k_map[0]][0, 0]], data[k_map[0]][:, 0])))
            y_interpolation = interpolate.interp1d(ts_pred, np.concatenate(([data[k_map[0]][0, 1]], data[k_map[0]][:, 1])))
            data[k_map[0]] = dict()
            data[k_map[0]]['x'] = x_interpolation
            data[k_map[0]]['y'] = y_interpolation

        # - for every prediction, use its timestamps for selecting the corresponding gt

        # # parse skeletons into numpy matrix [num_of_sk, num_of_keypoints, coords]
        # skeletons_shape = data[list(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys())[0]].shape
        # skeletons_gt = np.zeros((skeletons_shape[0], len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), skeletons_shape[1]))
        # for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
        #     skeletons_gt[:, k_map[1], :] = data[k_map[0]]

        head_sizes_gt_interp = interpolate.interp1d(ts_pred, np.concatenate(([data['head_sizes'][0]], data['head_sizes'])))
        torso_sizes_gt_interp = interpolate.interp1d(ts_pred, np.concatenate(([data['torso_sizes'][0]], data['torso_sizes'])))

        # TODO: check what size is present and name pck accordingly

        dataset_name = yarp_path.parent.parent.name
        results['datasets'][dataset_name] = dict()

        # parse predictions
        predictions_path = Path(args.predictions_path) / dataset_name
        predictions_file_path = list(predictions_path.glob('**/*.txt'))
        for pred_path in predictions_file_path:
            predictions = np.loadtxt(str(pred_path.resolve()), dtype=float)

            ts_pred = predictions[:, 0]
            skeletons_gt = np.zeros((len(ts_pred), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
            for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
                skeletons_gt[:, k_map[1], 0] = data[k_map[0]]['x'](ts_pred)
                skeletons_gt[:, k_map[1], 1] = data[k_map[0]]['y'](ts_pred)

            skeletons_pred = predictions[:, 2:].reshape(len(predictions), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)

            for metric_str in metrics:

                if metric_str == 'pck':
                    results['datasets'][dataset_name]['pck'] = dict()
                    for thi, th in enumerate(args.pck_thresholds):

                        # update dataset metric
                        # rms = metrics_utils.RMS()
                        # rms.update_samples(skeletons_pred, skeletons_gt)
                        # err = rms.get_value()

                        pck = metrics_utils.PCK(threshold=th)
                        pck.update_samples(skeletons_pred, skeletons_gt, torso_sizes_gt_interp(ts_pred))

                        algo_name = pred_path.stem
                        results['datasets'][dataset_name]['pck'][th] = dict()
                        results['datasets'][dataset_name]['pck'][th][algo_name] = pck

                        # update algo metric
                        if algo_name not in results['algos'].keys():
                            results['algos'][algo_name] = dict()

                        if 'pck' not in results['algos'][algo_name].keys():
                            results['algos'][algo_name]['pck'] = dict()

                        if th not in results['algos'][algo_name]['pck'].keys():
                            pck = metrics_utils.PCK(threshold=th)
                            results['algos'][algo_name]['pck'][th] = pck

                        pck = results['algos'][algo_name]['pck'][th]
                        pck.update_samples(skeletons_pred, skeletons_gt, torso_sizes_gt_interp(ts_pred))

                elif metric_str == 'rmse':
                    results['datasets'][dataset_name]['rmse'] = dict()
                    rmse = metrics_utils.RMSE()
                    rmse.update_samples(skeletons_pred, skeletons_gt)

                    algo_name = pred_path.stem
                    results['datasets'][dataset_name]['rmse'][algo_name] = rmse

                    # update algo metric
                    if algo_name not in results['algos'].keys():
                        results['algos'][algo_name] = dict()

                    if 'rmse' not in results['algos'][algo_name].keys():
                        rmse = metrics_utils.RMSE()
                        results['algos'][algo_name]['rmse'] = rmse

                    rmse = results['algos'][algo_name]['rmse']
                    rmse.update_samples(skeletons_pred, skeletons_gt)

                else:
                    print(f'metric {metric_str} not yet implemented')

            # TODO:
            # - sample random images from (good and bad?) predictions/gt list and plot joints

    # TODO:
    # - iterate over dictionary items and print metrics (generate tables in latex file?)
    # - output plots with graphs
    for (ds_name, ds) in results['datasets'].items():
        for (metric_name, metric_results) in ds.items():
            if metric_name == 'pck':
                # create table
                header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
                header.append('avg PCK')
                # header.insert(0, 'ds_name')

                for (th, algos) in metric_results.items():
                    table = list()
                    for (algo_name, metric) in algos.items():
                        joints_values, avg_value = metric.get_value()
                        table_row = joints_values.tolist()
                        table_row.append(avg_value)
                        table_row.insert(0, algo_name)
                        table.append(table_row)
                    print(f'PCK results for dataset {ds_name}, threshold {th}')
                    print(tabulate(table, headers=header))

            elif metric_name == 'rmse':
                header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in [f'{key} x', f'{key} y']]
                header.append('avg RMSE x')
                header.append('avg RMSE y')

                table = list()

                for (algo_name, metric) in metric_results.items():
                    joints_values, avg_values, max_values = metric.get_value()
                    table_row = joints_values.flatten().tolist()
                    table_row.extend(avg_values.flatten().tolist())
                    table_row.insert(0, algo_name)
                    table.append(table_row)
                print(f'RMSE results for dataset {ds_name}')
                print(tabulate(table, headers=header))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d', '--datasets_path', help='Path to the folders containing data saved in Yarp format', required=True)
    parser.add_argument('-p', '--predictions_path', help='Path to the predictions folder', required=True)
    parser.add_argument('-i', '--images_folder', help='Path to the folder containing the image frames')
    parser.add_argument('-o', '--output_folder', help='Path to the folder where evaluation results will be saved', required=True)
    parser.add_argument('-pck_th', '--pck_thresholds', help='List of thresholds for computing PCK', type=float, nargs='+', default=[.5])
    args = parser.parse_args()

    main(args)
