
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re

from pathlib import Path
from scipy import interpolate
from tabulate import tabulate
from typing import Optional

from datasets.utils import constants as ds_constants, parsing as ds_parsing
from evaluation.utils import metrics as metrics_utils, plots as plots_utils


# TODO:
#   - compute oks
#     - for a set of thresholds
#     - for each joint
#     - global
#   - randomly sample good and bad predictions and plot joints

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


def plot_pck_over_thresholds(pck_thresholds_results, output_folder_path, ds_name=None):

    # sort thresholds
    thresholds = list(pck_thresholds_results.keys())
    thresholds.sort()

    # group results by algorithm
    algos = dict()
    for th in thresholds:

        algo_metrics = pck_thresholds_results[th]
        for (algo_name, metric) in algo_metrics.items():

            if algo_name not in algos.keys():
                algos[algo_name] = list()

            values = metric.get_value()
            # joints_values = values[0]
            avg_value = values[1]

            algos[algo_name].append(avg_value * 100)

    # setup the figure
    my_dpi = 96
    if ds_name:
        figure_name = f'Dataset {ds_name}, Average PCK'
    else:
        figure_name = f'Global Average PCK'
    fig = plt.figure(num=figure_name,
                     figsize=(2048 / my_dpi, 900 / my_dpi),
                     dpi=96)
    ax = fig.add_subplot(111)
    fig.tight_layout(pad=5)

    # plot pcks
    for algo_name, pcks in algos.items():
        ax.plot(thresholds, pcks, color=np.random.rand(3, ), marker="o", label=f'{algo_name}',
                linestyle='-', alpha=1.0)

    # set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])

    # labels and title
    plt.xlabel('PCK thresholds', fontsize=22, labelpad=5)
    plt.ylabel(f'Average PCK %', fontsize=22, labelpad=5)
    fig.suptitle(figure_name, fontsize=28, y=0.97)
    plt.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=16, loc='upper right')

    # save plot
    if ds_name:
        fig_path = output_folder_path / f'{ds_name}_pck.png'
    else:
        fig_path = output_folder_path / f'global_pck.png'
    plt.savefig(str(fig_path.resolve()))


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


def tabulate_metric_over_algorithms(algo_metrics: dict, header: list, descr: Optional[str], to_latex: bool = False, file_path: Optional[Path] = None, file_stem: str = None) -> None:

    # create results table
    table = list()
    for (algo_name, metric) in algo_metrics.items():
        values = metric.get_value()
        joints_values = values[0]
        avg_value = values[1]
        table_row = joints_values.tolist()
        if isinstance(avg_value, np.ndarray):
            table_row.extend(avg_value.tolist())
        else:
            table_row.append(avg_value)
        table_row.insert(0, algo_name)
        table.append(table_row)

    # set table format
    fmt = 'simple'
    ext = '.txt'
    if to_latex:
        fmt = 'latex'
        ext = '.tex'

    if file_path and file_stem:

        full_path = file_path / (file_stem + ext)
        with open(str(full_path.resolve()), 'w') as f:

            # create latex string
            if to_latex:
                file_content = r'\documentclass[preview]{standalone}'\
                               r'\usepackage[utf8]{inputenc}'\
                               r'\usepackage{diagbox}'\
                               r'\begin{document}'\
                               r'\begin{table}'\
                               r'\centering'
                file_content += tabulate(table, headers=header, tablefmt=fmt)
                if descr:
                    file_content += r'\caption{' \
                                    f'{descr}' \
                                    r'}'
                file_content += r'\end{table}' \
                                r'\end{document}'

            else:
                file_content = tabulate(table, headers=header, tablefmt=fmt)

            f.write(file_content)

    else:
        if descr:
            print(descr)
        print(tabulate(table, headers=header, tablefmt=fmt))


def main(args):

    output_folder_path = Path(args.output_folder)
    output_folder_path = output_folder_path.resolve()
    output_folder_path.mkdir(parents=True, exist_ok=True)

    to_latex = args.latex

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

        ts_gt = np.concatenate(([.0], data['ts'], [data['ts'][-1] + 1]))

        # TODO: use numpy.interp
        # TODO: make it general for 2d/3d coordinates

        # interpolate ground truth joints so that they can be compared with the high frequency predictions
        for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
            x_interpolation = interpolate.interp1d(ts_gt, np.concatenate(([data[k_map[0]][0, 0]], data[k_map[0]][:, 0], [data[k_map[0]][-1, 0]])))
            y_interpolation = interpolate.interp1d(ts_gt, np.concatenate(([data[k_map[0]][0, 1]], data[k_map[0]][:, 1], [data[k_map[0]][-1, 1]])))
            data[k_map[0]] = dict()
            data[k_map[0]]['x'] = x_interpolation
            data[k_map[0]]['y'] = y_interpolation

        # GT contains the size of the torso
        if data['head_sizes'][0] == -1:
            pck_sizes_gt_interp = interpolate.interp1d(ts_gt, np.concatenate(([data['torso_sizes'][0]], data['torso_sizes'], [data['torso_sizes'][-1]])))
        # GT contains the size of the head
        else:
            pck_sizes_gt_interp = interpolate.interp1d(ts_gt, np.concatenate(([data['head_sizes'][0]], data['head_sizes'], [data['head_sizes'][-1]])))

        # ground truth in yarp format is supposed to be stored in folders name <dataset_name>/ch<channel_id>[frequency_info]skeleton
        # find the channel id
        numbers = re.findall('[0-9]+', yarp_path.parent.name)
        channel_id = numbers[0]
        parent_folder_prefix = yarp_path.parent.name.split(channel_id)[0]
        channel_folder = f'{parent_folder_prefix}{channel_id}'
        dataset_name = yarp_path.parent.parent.name
        results_key = f'{dataset_name}_{channel_folder}'

        output_ds_folder_path = output_folder_path / results_key
        output_ds_folder_path.mkdir(parents=True, exist_ok=True)

        results['datasets'][results_key] = dict()

        algorithm_names = []
        skeletons_predictions = []

        # parse predictions
        predictions_path = Path(args.predictions_path) / dataset_name / channel_folder
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
            if len(args.pck) != 0:

                if 'pck' not in results['datasets'][results_key].keys():
                    results['datasets'][results_key]['pck'] = dict()

                for thi, th in enumerate(args.pck):

                    # update dataset metric
                    pck = metrics_utils.PCK(threshold=th)
                    pck.update_samples(skeletons_pred, skeletons_gt, pck_sizes_gt_interp(ts_pred))

                    if th not in results['datasets'][results_key]['pck'].keys():
                        results['datasets'][results_key]['pck'][th] = dict()

                    results['datasets'][results_key]['pck'][th][algo_name] = pck

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

                if 'rmse' not in results['datasets'][results_key].keys():
                    results['datasets'][results_key]['rmse'] = dict()

                rmse = metrics_utils.RMSE()
                rmse.update_samples(skeletons_pred, skeletons_gt)

                algo_name = pred_path.stem
                results['datasets'][results_key]['rmse'][algo_name] = rmse

                # update algo metric
                if 'rmse' not in results['global'].keys():
                    results['global']['rmse'] = dict()

                if algo_name not in results['global']['rmse'].keys():
                    results['global']['rmse'][algo_name] = metrics_utils.RMSE()

                rmse = results['global']['rmse'][algo_name]
                rmse.update_samples(skeletons_pred, skeletons_gt)

        plot_predictions(output_ds_folder_path, results_key, ts_pred, skeletons_gt, algorithm_names, skeletons_predictions)

    # iterate over datasets metrics and print results
    for (ds_name, metrics) in results['datasets'].items():

        output_ds_folder_path = output_folder_path / ds_name
        output_ds_folder_path.mkdir(parents=True, exist_ok=True)

        for (metric_name, metric_results) in metrics.items():
            if metric_name == 'pck':

                plot_pck_over_thresholds(metric_results, output_ds_folder_path, ds_name)

                # create table
                header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
                header.append('avg PCK')

                for (th, algos) in metric_results.items():
                    tabulate_metric_over_algorithms(algos, header,
                                                    descr=f'PCK results for dataset {ds_name}, threshold {th}',
                                                    to_latex=to_latex,
                                                    file_path=output_ds_folder_path,
                                                    file_stem=f'pck_{th}_{ds_name}')

            elif metric_name == 'rmse':
                header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in [f'{key} x', f'{key} y']]
                header.append('avg RMSE x')
                header.append('avg RMSE y')

                tabulate_metric_over_algorithms(metric_results, header,
                                                descr=f'RMSE results for dataset {ds_name}',
                                                to_latex=to_latex,
                                                file_path=output_ds_folder_path,
                                                file_stem=f'rmse_{ds_name}')

    # iterate over global metrics and print results
    for (metric_name, metric_results) in results['global'].items():

        if metric_name == 'pck':

            plot_pck_over_thresholds(metric_results, output_folder_path)

            # create table
            header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
            header.append('avg PCK')

            for (th, algos) in metric_results.items():
                tabulate_metric_over_algorithms(algos, header,
                                                descr=f'Global PCK results for threshold {th}',
                                                to_latex=to_latex,
                                                file_path=output_folder_path,
                                                file_stem=f'pck_{th}')

        elif metric_name == 'rmse':
            # create table
            header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in [f'{key} x', f'{key} y']]
            header.append('avg RMSE x')
            header.append('avg RMSE y')

            tabulate_metric_over_algorithms(metric_results, header,
                                            descr=f'Global RMSE results',
                                            to_latex=to_latex,
                                            file_path=output_folder_path,
                                            file_stem='rmse')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d', '--datasets_path', help='Path to the folders containing data saved in Yarp format', required=True)
    parser.add_argument('-p', '--predictions_path', help='Path to the predictions folder', required=True)
    parser.add_argument('-i', '--images_folder', help='Path to the folder containing the image frames')
    parser.add_argument('-pck', help='List of thresholds for computing metric PCK; specifies that PCK must be computed', type=float, nargs='+', default=[])
    parser.add_argument('-rmse', help='flag specifying that the metric RMSE must be computed', dest='rmse', action='store_true')
    parser.set_defaults(rmse=False)
    parser.add_argument('-o', '--output_folder', help='Path to the folder where evaluation results will be saved', required=True)
    parser.add_argument('-latex', help='flag specifying that table results should saved to latex files', dest='latex', action='store_true')
    parser.set_defaults(latex=False)

    args = parser.parse_args()

    main(args)
