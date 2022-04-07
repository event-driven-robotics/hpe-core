
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy import interpolate
from tabulate import tabulate

from datasets.utils import constants as ds_constants, parsing as ds_parsing
from evaluation.utils import metrics as metrics_utils, plots as plots_utils


# TODO:
#   - plot pck auc
#   - compute oks
#     - for a set of thresholds
#     - for each joint
#     - global
#   - randomly sample good and bad predictions and plot joints
#   - add flag for generating latex tables

# class PredictionsPlot:
#
#     def __init__(self, output_folder_path, timestamps, joints_gt):
#
#         assert 1 <= joints_gt.shape[2] <= 3, 'coordinates must be either 2D or 3D'
#
#         self.output_folder_path = output_folder_path
#         self.timestamps = timestamps
#         self.joints_gt = joints_gt
#
#         # iterate on each joint
#         for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
#
#             for coord_ind in range(joints_gt.shape[2]):
#
#                 if coord_ind == 0:
#                     lbl_coord = 'x'
#                 if coord_ind == 1:
#                     lbl_coord = 'y'
#                 if coord_ind == 2:
#                     lbl_coord = 'z'
#
#                 lbl_legend_gt = lbl_coord + r'$_{GT}$'
#                 lbl_legend_pred = lbl_coord + r'$_{predicted}$'
#                 lbl_x_axis = 'time [sec]'
#                 lbl_y_axis = f'{lbl_coord} coordinate [px]'
#                 lbl_fig = f'Joint \'{joint_key}\', {lbl_coord} coordinate'
#
#                 # create plot
#                 my_dpi = 96
#                 fig = plt.figure(figsize=(2048 / my_dpi, 900 / my_dpi), dpi=96)
#                 ax = fig.add_subplot(111)
#                 fig.tight_layout(pad=5)
#
#                 # plot ground-truth
#                 coord_gt = joints_gt[:, joint_ind, coord_ind]
#                 ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.3, label=lbl_legend_gt)
#
#                 y_lim_min = math.inf
#                 y_lim_max = 0
#                 for predictions_algo in joints_predicted:
#                     # plot predictions
#                     coord_pred = predictions_algo[:, joint_ind, coord_ind]
#                     ax.plot(timestamps, coord_pred, color=np.random.rand(3, ), marker=".", label=lbl_legend_pred,
#                             linestyle='None', alpha=1.0)
#
#                     y_lim_min = min(y_lim_min, coord_pred)
#                     y_lim_max = max(y_lim_max, coord_pred)
#
#                 # set axis limits
#                 ax.set_xlim([timestamps[0], timestamps[-1]])
#                 ax.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])
#
#                 # labels and title
#                 plt.xlabel(lbl_x_axis, fontsize=22, labelpad=5)
#                 plt.ylabel(lbl_y_axis, fontsize=22, labelpad=5)
#                 fig.suptitle(lbl_fig, fontsize=28, y=0.97)
#                 plt.tick_params(axis='both', which='major', labelsize=18)
#                 ax.legend(fontsize=16, loc='upper right')


def plot_predictions(output_folder_path, ds_name, timestamps, joints_gt, algo_names, joints_predicted):

    assert 1 <= joints_gt.shape[2] <= 3, 'coordinates must be either 2D or 3D'

    # iterate on each joint
    for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():

        for coord_ind in range(joints_gt.shape[2]):

            if coord_ind == 0:
                lbl_coord = 'x'
            if coord_ind == 1:
                lbl_coord = 'y'
            if coord_ind == 2:
                lbl_coord = 'z'

            # create plot
            my_dpi = 96
            fig = plt.figure(num=f'Dataset {ds_name}, Joint \'{joint_key}\', {lbl_coord} coordinate',
                             figsize=(2048 / my_dpi, 900 / my_dpi),
                             dpi=96)
            ax = fig.add_subplot(111)
            fig.tight_layout(pad=5)

            # plot ground-truth
            coord_gt = joints_gt[:, joint_ind, coord_ind]
            ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.3, label='GT')

            y_lim_min = min(coord_gt)
            y_lim_max = max(coord_gt)
            for pi, predictions_algo in enumerate(joints_predicted):

                # plot predictions
                coord_pred = predictions_algo[:, joint_ind, coord_ind]
                ax.plot(timestamps, coord_pred, color=np.random.rand(3,), marker=".", label=f'{algo_names[pi]}', linestyle='None', alpha=1.0)

                y_lim_min = min(y_lim_min, min(coord_pred))
                y_lim_max = max(y_lim_max, max(coord_pred))

            # set axis limits
            ax.set_xlim([timestamps[0], timestamps[-1]])
            ax.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])

            # labels and title
            plt.xlabel('time [sec]', fontsize=22, labelpad=5)
            plt.ylabel(f'{lbl_coord} coordinate [px]', fontsize=22, labelpad=5)
            fig.suptitle(f'dataset {ds_name}, joint \'{joint_key}\', {lbl_coord} coordinate', fontsize=28, y=0.97)
            plt.tick_params(axis='both', which='major', labelsize=18)
            ax.legend(fontsize=16, loc='upper right')

            # save plot
            fig_path = output_folder_path / f'{ds_name}_{joint_key}_predictions_{lbl_coord}.png'
            plt.savefig(str(fig_path.resolve()))


def main(args):

    output_folder_path = Path(args.output_folder)
    output_folder_path = output_folder_path.resolve()
    output_folder_path.mkdir(parents=True, exist_ok=True)

    results = dict()
    results['datasets'] = dict()
    results['global'] = dict()

    # import GT from yarp
    datasets_path = Path(args.datasets_path)
    yarp_file_paths = list(datasets_path.glob('**/data.log'))
    for yarp_path in yarp_file_paths:

        if 'skeleton' not in yarp_path.parent.name:
            continue

        data = ds_parsing.import_yarp_skeleton_data(yarp_path)

        ts_pred = np.concatenate(([.0], data['ts']))

        # TODO: use numpy.interp
        # TODO: make it general for 2d/3d coordinates

        # interpolate ground truth joints so that they can be compared with the high frequency predictions
        for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
            x_interpolation = interpolate.interp1d(ts_pred, np.concatenate(([data[k_map[0]][0, 0]], data[k_map[0]][:, 0])))
            y_interpolation = interpolate.interp1d(ts_pred, np.concatenate(([data[k_map[0]][0, 1]], data[k_map[0]][:, 1])))
            data[k_map[0]] = dict()
            data[k_map[0]]['x'] = x_interpolation
            data[k_map[0]]['y'] = y_interpolation

        # GT contains the size of the torso
        if data['head_sizes'][0] == -1:
            pck_sizes_gt_interp = interpolate.interp1d(ts_pred, np.concatenate(([data['torso_sizes'][0]], data['torso_sizes'])))
        # GT contains the size of the head
        else:
            pck_sizes_gt_interp = interpolate.interp1d(ts_pred, np.concatenate(([data['head_sizes'][0]], data['head_sizes'])))

        dataset_name = yarp_path.parent.parent.name

        output_ds_folder_path = output_folder_path / dataset_name
        output_ds_folder_path.mkdir(parents=True, exist_ok=True)

        results['datasets'][dataset_name] = dict()

        algorithm_names = []
        skeletons_predictions = []

        # parse predictions
        predictions_path = Path(args.predictions_path) / dataset_name
        predictions_file_path = list(predictions_path.glob('**/*.txt'))
        for pred_path in predictions_file_path:

            algo_name = pred_path.stem

            predictions = np.loadtxt(str(pred_path.resolve()), dtype=float)

            ts_pred = predictions[:, 0]
            skeletons_gt = np.zeros((len(ts_pred), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
            for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
                skeletons_gt[:, k_map[1], 0] = data[k_map[0]]['x'](ts_pred)
                skeletons_gt[:, k_map[1], 1] = data[k_map[0]]['y'](ts_pred)

            skeletons_pred = predictions[:, 2:].reshape(len(predictions), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)

            algorithm_names.append(algo_name)
            skeletons_predictions.append(skeletons_pred)

            # compute PCK
            if not len(args.pck) == 0:
                results['datasets'][dataset_name]['pck'] = dict()
                for thi, th in enumerate(args.pck):

                    # update dataset metric
                    pck = metrics_utils.PCK(threshold=th)
                    pck.update_samples(skeletons_pred, skeletons_gt, pck_sizes_gt_interp(ts_pred))

                    results['datasets'][dataset_name]['pck'][th] = dict()
                    results['datasets'][dataset_name]['pck'][th][algo_name] = pck

                    # update global metric
                    if 'pck' not in results['global'].keys():
                        results['global']['pck'] = dict()

                    if th not in results['global']['pck'].keys():
                        results['global']['pck'][th] = dict()

                    if algo_name not in results['global']['pck'][th].keys():
                        results['global']['pck'][th][algo_name] = metrics_utils.PCK(threshold=th)

                    pck = results['global']['pck'][th][algo_name]
                    pck.update_samples(skeletons_pred, skeletons_gt, pck_sizes_gt_interp(ts_pred))

            # compute RMSE
            if args.rmse:
                results['datasets'][dataset_name]['rmse'] = dict()
                rmse = metrics_utils.RMSE()
                rmse.update_samples(skeletons_pred, skeletons_gt)

                algo_name = pred_path.stem
                results['datasets'][dataset_name]['rmse'][algo_name] = rmse

                # update algo metric
                if 'rmse' not in results['global'].keys():
                    results['global']['rmse'] = dict()

                if algo_name not in results['global']['rmse'].keys():
                    results['global']['rmse'][algo_name] = metrics_utils.RMSE()

                rmse = results['global']['rmse'][algo_name]
                rmse.update_samples(skeletons_pred, skeletons_gt)

        plot_predictions(output_ds_folder_path, dataset_name, ts_pred, skeletons_gt, algorithm_names, skeletons_predictions)

    # iterate over datasets metrics and print results
    for (ds_name, metrics) in results['datasets'].items():
        for (metric_name, metric_results) in metrics.items():
            if metric_name == 'pck':
                # create table
                header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
                header.append('avg PCK')
                # header.insert(0, 'ds_name')

                for (th, algos) in metric_results.items():
                    table = list()
                    for (algo_name, metrics) in algos.items():
                        joints_values, avg_value = metrics.get_value()
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

                for (algo_name, metrics) in metric_results.items():
                    joints_values, avg_values, max_values = metrics.get_value()
                    table_row = joints_values.flatten().tolist()
                    table_row.extend(avg_values.flatten().tolist())
                    table_row.insert(0, algo_name)
                    table.append(table_row)
                print(f'RMSE results for dataset {ds_name}')
                print(tabulate(table, headers=header))

    # iterate over global metrics and print results
    for (metric_name, metric_results) in results['global'].items():

        if metric_name == 'pck':
            # create table
            header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
            header.append('avg PCK')

            for (th, algos) in metric_results.items():
                table = list()
                for (algo_name, metrics) in algos.items():
                    joints_values, avg_value = metrics.get_value()
                    table_row = joints_values.tolist()
                    table_row.append(avg_value)
                    table_row.insert(0, algo_name)
                    table.append(table_row)
                print(f'Global PCK results for threshold {th}')
                print(tabulate(table, headers=header))

        elif metric_name == 'rmse':
            # create table
            header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in [f'{key} x', f'{key} y']]
            header.append('avg RMSE x')
            header.append('avg RMSE y')

            table = list()

            for (algo_name, metrics) in metric_results.items():
                joints_values, avg_values, max_values = metrics.get_value()
                table_row = joints_values.flatten().tolist()
                table_row.extend(avg_values.flatten().tolist())
                table_row.insert(0, algo_name)
                table.append(table_row)
            print(f'Global RMSE results')
            print(tabulate(table, headers=header))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d', '--datasets_path', help='Path to the folders containing data saved in Yarp format', required=True)
    parser.add_argument('-p', '--predictions_path', help='Path to the predictions folder', required=True)
    parser.add_argument('-i', '--images_folder', help='Path to the folder containing the image frames')
    parser.add_argument('-o', '--output_folder', help='Path to the folder where evaluation results will be saved', required=True)
    parser.add_argument('-pck', help='List of thresholds for computing metric PCK; specifies that PCK must be computed', type=float, nargs='+', default=[])
    parser.add_argument('-rmse', help='flag specifying the metric RMSE must be computed', dest='rmse', action='store_true')
    parser.set_defaults(rmse=False)

    args = parser.parse_args()

    main(args)