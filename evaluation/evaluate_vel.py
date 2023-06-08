
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
        # ax.plot(thresholds, pcks, color=np.random.rand(3, ), marker="o", label=f'{algo_name}',
        #         linestyle='-', alpha=1.0)
        ax.plot(thresholds, pcks, marker="o", label=f'{algo_name}',
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


def plot_boxplot(algo_metrics: dict, descr: Optional[str], file_path: Path) -> None:

    # create figure
    fig, ax = plt.subplots()

    all_values = list()
    tick_labels = [' ']
    for (algo_name, metric) in sorted(algo_metrics.items(), reverse = True):
        values = metric.get_value()
        joints_metric_values = values[0]

        if isinstance(metric, metrics_utils.PCK):
            all_values.append(joints_metric_values)

            tick_labels.append(algo_name)
            ax.set_xlabel('Joints PCK', fontsize = 22)

        elif isinstance(metric, metrics_utils.RMSE):
            #all_values.append(joints_metric_values[::2])  # add metric values for x coordinates
            #all_values.append(joints_metric_values[1::2])  # add metric values for y coordinates
            all_values.append(np.maximum(joints_metric_values[::2], joints_metric_values[1::2]))

            tick_labels.extend([f'{algo_name}'])
            ax.set_xlabel('Joints RMSE', fontsize = 22)
        
        elif isinstance(metric, metrics_utils.MPJPE):
            all_values.append(joints_metric_values)

            tick_labels.extend([f'{algo_name}'])
            ax.set_xlabel('Joints MPJPE', fontsize = 22)

    # plot values
    y_ticks = np.arange(len(tick_labels))
    meanlineprops = dict(linestyle='-', linewidth=2.0, color='tab:blue')
    ax.boxplot(all_values, vert=False, showfliers=False, showmeans=True, meanline=True, meanprops=meanlineprops)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(tick_labels, fontsize = 18)
    ax.set_title(descr, fontsize = 24)
    plt.xticks(fontsize=18, rotation=0)
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(str(file_path.resolve()))
    

def plot_error(algo_metrics: dict, headers: list, descr: Optional[str], file_path: Path) -> None:

    # create figure
    fig, ax = plt.subplots()

    all_values = list()
    tick_labels = [' ']
    for (algo_name, metric) in sorted(algo_metrics.items(), reverse = True):
        values = metric.get_value()
        joints_metric_values = values[0]

        if isinstance(metric, metrics_utils.PCK):
            all_values.append(joints_metric_values)

            tick_labels.append(algo_name)
            ax.set_xlabel('Joints PCK', fontsize = 22)

        elif isinstance(metric, metrics_utils.RMSE):
            all_values.append(np.maximum(joints_metric_values[::2], joints_metric_values[1::2]))

            tick_labels.extend([f'{algo_name}'])
            ax.set_xlabel('Joints RMSE', fontsize = 22)
            
            res = np.maximum(joints_metric_values[::2], joints_metric_values[1::2])

        
        elif isinstance(metric, metrics_utils.MPJPE):
            all_values.append(joints_metric_values)
            res = joints_metric_values

            tick_labels.extend([f'{algo_name}'])
            ax.set_xlabel('Joints MPJPE', fontsize = 22)

        
        ax.plot(res, label=f'{algo_name}', marker='o', markersize=10)
    plt.xticks(range(0, 13), tuple(headers[0:13]), fontsize=20, rotation=35, ha='right')
    ax.legend(fontsize=22, loc='upper left')
    plt.xlabel('MPJPE [px]', fontsize=24, labelpad=5)
    plt.tick_params(axis='y', which='major', labelsize=22)


def plot_predictions(output_folder_path, ds_name, timestamps, joints_gt, algo_names, joints_predicted):

    assert 1 <= joints_gt.shape[2] <= 3, 'coordinates must be either 2D or 3D'

    # define a color for each algorithm and each coordinate
    # algo_colors = [[np.random.rand(3,) for _ in range(joints_gt.shape[2])] for _ in algo_names] # random colors
    algo_colors = [['tab:green','tab:orange'], ['blue', 'red'], ['violet', 'chocolate'], ['purple', 'sienna'], ['lime', 'gold'], ['aqua', 'olive'], ['tab:blue', 'green'], ['gold', 'lime']] # fixed colors

    # # iterate on each joint
    # for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
    #     # create plot
    #     my_dpi = 96
    #     fig = plt.figure(num=f'Dataset {ds_name}, Joint \'{joint_key}\'',
    #                       figsize=(2048 / my_dpi, 900 / my_dpi),
    #                       dpi=96)
    #     ax = fig.add_subplot(111)
    #     fig.tight_layout(pad=5)

    #     y_lim_min = np.inf
    #     y_lim_max = 0

    #     for coord_ind in range(joints_gt.shape[2]):

    #         # plot ground-truth
    #         coord_gt = joints_gt[:, joint_ind, coord_ind]
    #         if coord_ind == 0:
    #             lbl_coord = 'x'
    #             ax.plot(timestamps, coord_gt, color='tab:blue', alpha=0.2, label=f'GT {lbl_coord}')
    #         if coord_ind == 1:
    #             lbl_coord = 'y'
    #             ax.plot(timestamps, coord_gt, color='tab:orange', alpha=0.2, label=f'GT {lbl_coord}')
    #         if coord_ind == 2:
    #             lbl_coord = 'z'
    #             ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.2, label=f'GT {lbl_coord}')



    #         y_lim_min = min(y_lim_min, min(coord_gt))
    #         y_lim_max = max(y_lim_max, max(coord_gt))

    #         for pi, predictions_algo in enumerate(joints_predicted):

    #             # plot predictions
    #             coord_pred = predictions_algo[:, joint_ind, coord_ind]
    #             ax.plot(timestamps, coord_pred, color=algo_colors[pi][coord_ind], marker=".", label=f'{algo_names[pi][1:]} {lbl_coord}', linestyle='-', alpha=1.0, markersize = 12)
    #             y_lim_min = min(y_lim_min, min(coord_pred))
    #             y_lim_max = max(y_lim_max, max(coord_pred))

    #         # set axis limits
    #         ax.set_xlim([timestamps[0], timestamps[-1]])
    #         ax.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])
   

    #         # labels and title
    #         plt.xlabel('time [sec]', fontsize=22, labelpad=5)
    #         plt.ylabel('coordinates [px]', fontsize=22, labelpad=5)
    #         fig.suptitle(f'dataset {ds_name}, joint \'{joint_key}\' coordinates', fontsize=28, y=0.97)
    #         plt.tick_params(axis='both', which='major', labelsize=18)
    #         ax.legend(fontsize=16, loc='upper right')

    #         # save plot
    #         fig_path = output_folder_path / f'{ds_name}_{joint_key}_predictions.png'
    #         # plt.savefig(str(fig_path.resolve()))
    
    
    # plot for a single joint
    # joint_key = 'head'
    # joint_ind = 0
    
    # iterate on each joint
    for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
        # create plot
        my_dpi = 96
        fig = plt.figure(num=f'Dataset {ds_name}, Joint \'{joint_key}\'',
                          figsize=(2048 / my_dpi, 900 / my_dpi),
                          dpi=96)
        ax1 = plt.subplot(211)
        plt.xlabel('time [sec]', fontsize=24, labelpad=5)
        plt.ylabel('vx [px/sex]', fontsize=24, labelpad=5)
        plt.tick_params(axis='both', which='major', labelsize=18)
        ax2 = plt.subplot(212)
        plt.xlabel('time [sec]', fontsize=24, labelpad=5)
        plt.ylabel('vx [px/sex]', fontsize=24, labelpad=5)
        plt.tick_params(axis='both', which='major', labelsize=18)
        fig.tight_layout(pad=5)
        
        
        y_lim_min = np.inf
        y_lim_max = 0
    
        for coord_ind in range(joints_gt.shape[2]):
    
            # plot ground-truth
            coord_gt = joints_gt[:, joint_ind, coord_ind]
            if coord_ind == 0:
                lbl_coord = 'x'
                ax1.plot(timestamps, coord_gt, color='tab:blue', alpha=0.5, label=f'GT {lbl_coord}')
            if coord_ind == 1:
                lbl_coord = 'y'
                ax2.plot(timestamps, coord_gt, color='tab:purple', alpha=0.5, label=f'GT {lbl_coord}')
            # if coord_ind == 2:
            #     lbl_coord = 'z'
            #     ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.5, label=f'GT {lbl_coord}')
    
    
    
            y_lim_min = min(y_lim_min, min(coord_gt))
            y_lim_max = max(y_lim_max, max(coord_gt))
    
            for pi, predictions_algo in enumerate(joints_predicted):
                # plot predictions
                coord_pred = predictions_algo[:, joint_ind, coord_ind]
                if coord_ind == 0:
                   ax1.plot(timestamps, coord_pred, color=algo_colors[pi][coord_ind], marker="None", label=f'{algo_names[pi][0:]} {lbl_coord}', linestyle='-', alpha=1.0, markersize = 16)
                if coord_ind == 1:  
                    ax2.plot(timestamps, coord_pred, color=algo_colors[pi][coord_ind], marker="None", label=f'{algo_names[pi][0:]} {lbl_coord}', linestyle='-', alpha=1.0, markersize = 16)
                y_lim_min = min(y_lim_min, min(coord_pred))
                y_lim_max = max(y_lim_max, max(coord_pred))
    
            # set axis limits
            ax1.set_xlim([timestamps[0], timestamps[-1]])
            # ax1.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])
            ax1.set_ylim([-100, 100])
            
            ax2.set_xlim([timestamps[0], timestamps[-1]])
            # ax2.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])
            ax2.set_ylim([-100, 100])
       
    
            # labels and title
    
            fig.suptitle(f'dataset {ds_name}, joint \'{joint_key}\' coordinates', fontsize=28, y=0.97)
            
            ax1.legend(fontsize=20, loc='upper right')
            ax2.legend(fontsize=20, loc='upper right')
    
            # save plot
            fig_path = output_folder_path / f'{ds_name}_{joint_key}_predictions.png'
            # plt.savefig(str(fig_path.resolve()))
            

def plot_predictions_all_joints(output_folder_path, ds_name, timestamps, joints_gt, algo_names, joints_predicted):

    assert 1 <= joints_gt.shape[2] <= 3, 'coordinates must be either 2D or 3D'

    for pi, predictions_algo in enumerate(joints_predicted):
        # create plot
        my_dpi = 96
        fig = plt.figure(num=f'Dataset {ds_name} - all joints - Algorithm: {algo_names[pi]}',
                         figsize=(2048 / my_dpi, 900 / my_dpi),
                         dpi=96)
        ax = fig.add_subplot(111)
        fig.tight_layout(pad=5)
    
        y_lim_min = np.inf
        y_lim_max = 0
        
        # iterate on each joint
        for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
            for coord_ind in range(joints_gt.shape[2]):
    
                # plot ground-truth
                coord_gt = joints_gt[:, joint_ind, coord_ind]
                if coord_ind == 0:
                    lbl_coord = 'x'
                    if(joint_ind==0):
                        ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.8, label=f'GT {lbl_coord}')
                    ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.8)
                if coord_ind == 1:
                    lbl_coord = 'y'
                    if(joint_ind==0):
                        ax.plot(timestamps, coord_gt, color='red', alpha=0.8, label=f'GT {lbl_coord}')
                    ax.plot(timestamps, coord_gt, color='red', alpha=0.8)
                if coord_ind == 2:
                    lbl_coord = 'z'
                    ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.2, label=f'GT {lbl_coord}')
    
    
                y_lim_min = min(y_lim_min, min(coord_gt))
                y_lim_max = max(y_lim_max, max(coord_gt))
    

                # plot predictions
                coord_pred = predictions_algo[:, joint_ind, coord_ind]
                if(coord_ind==0): # x component
                    if(joint_ind==0):
                        ax.plot(timestamps, coord_pred, color='tab:blue', marker=".", linestyle='-', alpha=1.0, label=f'Tracked {lbl_coord}')
                    ax.plot(timestamps, coord_pred, color='tab:blue', marker=".", linestyle='-', alpha=1.0)
                if(coord_ind==1): # y component
                    if(joint_ind==0):
                        ax.plot(timestamps, coord_pred, color='tab:orange', marker=".", linestyle='-', alpha=1.0, label=f'Tracked {lbl_coord}')
                    ax.plot(timestamps, coord_pred, color='tab:orange', marker=".", linestyle='-', alpha=1.0)

                y_lim_min = min(y_lim_min, min(coord_pred))
                y_lim_max = max(y_lim_max, max(coord_pred))
    
        # set axis limits
        ax.set_xlim([timestamps[0], timestamps[-1]])
        ax.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])
    
        # labels and title
        plt.xlabel('time [sec]', fontsize=22, labelpad=5)
        plt.ylabel('coordinates [px]', fontsize=22, labelpad=5)
        fig.suptitle(f'Dataset {ds_name} - all joints - Algorithm: {algo_names[pi]}', fontsize=28, y=0.97)
        plt.tick_params(axis='both', which='major', labelsize=18)
        ax.legend(fontsize=16, loc='upper right')
    
        # save plot
        fig_path = output_folder_path / f'{ds_name}_{joint_key}_predictions.png'
        # plt.savefig(str(fig_path.resolve()))
            
            

def plot_latency(output_folder_path, ds_name, timestamps, algo_names, latencies):

    # define a color for each algorithm and each coordinate
    # algo_colors = [[np.random.rand(3,) for _ in range(joints_gt.shape[2])] for _ in algo_names] # random colors
    algo_colors = ['tab:green','tab:orange','blue', 'red', 'violet', 'chocolate', 'purple', 'sienna', 'lime', 'gold', 'aqua', 'olive'] # fixed colors

    

    # create plot
    my_dpi = 96
    fig = plt.figure(num=f'Latencies: Dataset {ds_name}',
                     figsize=(2048 / my_dpi, 900 / my_dpi),
                     dpi=96)
    ax = fig.add_subplot(111)
    fig.tight_layout(pad=5)

    y_lim_min = np.inf
    y_lim_max = 0


    for pi, latency in enumerate(latencies):
        # plot predictions
        coord_pred = latency
        t_axis = timestamps[pi]
        # ax.plot(timestamps[pi], coord_pred, color=algo_colors[pi], marker="None", label=f'{algo_names[pi]}', linestyle='-', alpha=1.0)
        ax.plot(timestamps[pi], coord_pred,  marker="None", label=f'{algo_names[pi]}', linestyle='-', alpha=1.0)

        y_lim_min = min(y_lim_min, min(latency))
        y_lim_max = max(y_lim_max, max(latency))
    # ax.plot(timestamps, latencies, marker=".", linestyle='-', alpha=1.0)

    # set axis limits
    ax.set_xlim([min([min(r) for r in timestamps]), max([max(r) for r in timestamps])])
    ax.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])

    # labels and title
    plt.xlabel('time [sec]', fontsize=22, labelpad=5)
    plt.ylabel('Latency of detection [msec]', fontsize=22, labelpad=5)
    fig.suptitle(f'Latencies for dataset {ds_name}', fontsize=28, y=0.97)
    plt.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=16, loc='upper right')

    # save plot
    # fig_path = output_folder_path / f'{ds_name}_{joint_key}_predictions.png'
    # plt.savefig(str(fig_path.resolve()))



def tabulate_metric_over_algorithms(algo_metrics: dict, headers: list, descr: Optional[str], to_latex: bool = False, file_path: Optional[Path] = None, file_stem: str = None) -> None:

    # create results table
    table = list()
    header = list()
    header.append("Joint")
    table.append(headers)
    for (algo_name, metric) in sorted(algo_metrics.items()):
        values = metric.get_value()
        if(len(values)>2):
            joints_values = np.around(values[2], decimals=2)
        else:
            joints_values = np.around(values[0], decimals=2)
        # joints_values = values[0]
        # avg_value = np.max(values[1])
        avg_value =  np.around(np.mean(joints_values), decimals=2)
        # joints_values = values[0] # this mean matches the mean in the data in boxplot (green line), not the median (orange line)
        table_row = joints_values.tolist()
        if isinstance(avg_value, np.ndarray):
            table_row.extend(avg_value.tolist())
        else:
            table_row.append(avg_value)
        table.append(table_row)
        header.append(algo_name)
    # transpose list of lists
    table = list(map(list, zip(*table)))
        

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
                file_content = ''
                file_content += r"""\documentclass[varwidth]{standalone}
\usepackage[utf8]{inputenc}
\begin{document}
\begin{table}
\centering"""
                file_content += tabulate(table, headers=header, tablefmt=fmt)
                if descr:
                    descr_tex = descr.replace('_', '\_')
                    file_content += r'\caption{' \
                                    f'{descr_tex}' \
                                    r'}'
                file_content += r"""
\end{table}
\end{document}"""

            else:
                file_content = tabulate(table, headers=header, tablefmt=fmt)
                print(file_content)

            f.write(file_content)

    else:
        if descr:
            print(descr)
        print(tabulate(table, headers=header, tablefmt=fmt))
        
        
def tabulate_latency_over_algorithms(algo_metrics: dict, headers: list, descr: Optional[str], to_latex: bool = False, file_path: Optional[Path] = None, file_stem: str = None) -> None:

    # create results table
    table = list()
    header = list()
    header.append("")
    table.append(headers)
    for (algo_name, metric) in sorted(algo_metrics.items()):
        table_row = []
        table_row.append( metric.tolist())
        table.append(table_row)
        header.append(algo_name)
    # transpose list of lists
    table = list(map(list, zip(*table)))
        

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
                file_content = ''
                file_content += r"""\documentclass[varwidth]{standalone}
\usepackage[utf8]{inputenc}
\begin{document}
\begin{table}
\centering"""
                file_content += tabulate(table, headers=header, tablefmt=fmt)
                if descr:
                    descr_tex = descr.replace('_', '\_')
                    file_content += r'\caption{' \
                                    f'{descr_tex}' \
                                    r'}'
                file_content += r"""
\end{table}
\end{document}"""

            else:
                file_content = tabulate(table, headers=header, tablefmt=fmt)
                print(file_content)

            f.write(file_content)

    else:
        if descr:
            print(descr)
        print(tabulate(table, headers=header, tablefmt=fmt))

def plot_skeleton(output_folder_path, ds_name, timestamps, joints_gt, algo_names, joints_predicted, vel_gt1K):

    eros = plt.imread("/home/fdipietro/eros.png")
    evs = plt.imread("/home/fdipietro/evs.png")
    # fig, ax = plt.subplots()
    
    assert 1 <= joints_gt.shape[2] <= 3, 'coordinates must be either 2D or 3D'
    algo_colors = ['tab:green','blue', 'red', 'violet', 'chocolate', 'purple', 'sienna', 'lime', 'gold', 'aqua', 'olive'] # fixed colors

    # for pi, predictions_algo in enumerate(joints_predicted):
    # create plot
    my_dpi = 96
    # fig = plt.figure(num=f'Dataset {ds_name} - all joints - Algorithm: {algo_names[pi]}',
    #                  figsize=(2048 / my_dpi, 900 / my_dpi),
    #                  dpi=96)
    fig = plt.figure(num=f'Dataset {ds_name} - all joints',
                     figsize=(2048 / my_dpi, 900 / my_dpi),
                     dpi=96)
    ax = fig.add_subplot(111)
    fig.tight_layout(pad=5)
    # ax.imshow(evs, alpha=1) # IMPORTANT UNCOMMENT THIS LINE
    # ax.imshow(eros, alpha=1)
    ts = 26 # selected timestamp in seconds
    ts = ts * 1000
    
    # GROUND-TRUTH
    x = np.array([])
    y = np.array([])
    vx = np.array([])
    vy = np.array([])
    sc = 3
    

    # iterate on each joint
    for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
        x0 = joints_gt[ts, joint_ind,0]
        vx0 = vel_gt1K[ts, joint_ind,0] * sc
        y0 = 480-joints_gt[ts, joint_ind,1]
        vy0 = vel_gt1K[ts, joint_ind,1] * sc
        x = np.append(x, x0)
        y = np.append(y, y0)
        vx = np.append(vx, vx0)
        vy = np.append(vy, vy0)
        ax.arrow(x0, y0, vx0, vy0, head_width=5, head_length=10, color="tab:red")
        # ax.plot([xp0], [yp0], color='tab:orange', marker='x')
    # ax.plot(0, 0, color="tab:orange",  label=f'Ground-truth velocity')
    print(vx, vy)
    # plot joints location
    ax.scatter(x, y)
    ax.set_aspect(4/3)
    # add segments bewtween joints
    ax.plot([x[1], x[2]], [y[1], y[2]], color = 'tab:blue', linewidth=3) #shoulderR - shoulderL
    ax.plot([x[1], x[3]], [y[1], y[3]], color = 'tab:blue', linewidth=3) #shoulderR - elbowR
    ax.plot([x[2], x[4]], [y[2], y[4]], color = 'tab:blue', linewidth=3) #shoulderL - elbowL
    ax.plot([x[2], x[5]], [y[2], y[5]], color = 'tab:blue', linewidth=3) #shoulderL - wristL
    ax.plot([x[1], x[6]], [y[1], y[6]], color = 'tab:blue', linewidth=3) #shoulderR - wristR
    ax.plot([x[3], x[7]], [y[3], y[7]], color = 'tab:blue', linewidth=3) #elbowR - handR
    ax.plot([x[4], x[8]], [y[4], y[8]], color = 'tab:blue', linewidth=3) #elbowL - handL
    ax.plot([x[6], x[5]], [y[5], y[5]], color = 'tab:blue', linewidth=3) #wristR - wristL
    ax.plot([x[6], x[9]], [y[6], y[9]], color = 'tab:blue', linewidth=3) #wristR - kneeR
    ax.plot([x[5], x[10]], [y[5], y[10]], color = 'tab:blue', linewidth=3) #wristL - kneeL
    ax.plot([x[9], x[11]], [y[9], y[11]], color = 'tab:blue', linewidth=3) #kneeR - footR
    ax.plot([x[10], x[12]], [y[10], y[12]], color = 'tab:blue', linewidth=3) #kneeL - footL
    
    circleHead = plt.Circle((x[0],y[0]),20, fill=False, color = 'tab:blue', linewidth=3)
    ax.add_patch(circleHead)
    ax.plot([(x[1]+x[2])/2 , x[0]], [(y[1]+y[2])/2 , y[0]-20], color = 'tab:blue', linewidth=3) #shoulderR - shoulderL
    
    ax.set_xlim([0, 639])
    ax.set_ylim([0, 479])  
    
    # for pi, predictions_algo in enumerate(joints_predicted):
    #     for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
    #         x = np.array([])
    #         y = np.array([])
    #         vx = np.array([])
    #         vy = np.array([])
    #         x0 = joints_gt[ts, joint_ind,0]
    #         vx0 = predictions_algo[ts, joint_ind,0] * sc
    #         y0 = 480-joints_gt[ts, joint_ind,1]
    #         vy0 = predictions_algo[ts, joint_ind,1] * sc
    #         x = np.append(x, x0)
    #         y = np.append(y, y0)
    #         vx = np.append(vx, vx0)
    #         vy = np.append(vy, vy0)
    #         ax.arrow(x0, y0, vx0, vy0, head_width=5, head_length=10, color=algo_colors[pi])
    #     ax.plot(0, 0, color=algo_colors[pi],  label=f'Estimated velocity with  {algo_names[pi]}')
    # ax.legend(fontsize=16, loc='upper right')       


def main(args):
    
    plt.close('all')

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

        # ground truth in yarp format is supposed to be stored in folders name <dataset_name>/ch<channel_id>[frequency_info]skeleton
        # find the channel id
        numbers = re.findall('[0-9]+', yarp_path.parent.name)
        channel_id = numbers[0]
        parent_folder_prefix = yarp_path.parent.name.split(channel_id)[0]
        channel_folder = f'{parent_folder_prefix}{channel_id}'
        dataset_name = yarp_path.parent.parent.name
        results_key = f'{dataset_name}_{channel_folder}'

        predictions_path = Path(args.predictions_path) / dataset_name
        predictions_file_path = list(predictions_path.glob('**/*.csv'))
        predictions_file_path.sort()
        if len(predictions_file_path) == 0:
            print('\x1b[1;33;20m' + "Skipping " + str(dataset_name) + " as no results exist in" + str(predictions_path) + '\x1b[0m')
            continue
        
        predictions_file_path.sort(reverse=True)
        data = ds_parsing.import_yarp_skeleton_data(yarp_path)

        ts_gt = np.concatenate(([.0], data['ts'], [data['ts'][-1] + 1]))
        
        # TODO: use numpy.interp
        # TODO: make it GeneratorExit()eral for 2d/3d coordinates

        # interpolate ground truth joints so that they can be compared with the high frequency predictions
        for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
            x_interpolation = interpolate.interp1d(ts_gt, np.concatenate(([data[k_map[0]][0, 0]], data[k_map[0]][:, 0], [data[k_map[0]][-1, 0]])))#, fill_value="extrapolate"
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

        output_ds_folder_path = output_folder_path / results_key
        output_ds_folder_path.mkdir(parents=True, exist_ok=True)

        results['datasets'][results_key] = dict()

        algorithm_names = []
        skeletons_predictions = []
        skeletons_predictions1K = []
        timestamps = []
        latency = []

        # parse predictions
        for pred_path in predictions_file_path:

            algo_name = pred_path.stem
            predictions_old = np.loadtxt(str(pred_path.resolve()), dtype=float)

            # ts_pred = predictions[:, 0]
            idx = np.where(np.logical_and(predictions_old[:, 0]>0.1*ts_gt[-1],predictions_old[:, 0]<ts_gt[-1]))
            # idx = np.where(predictions_old[:, 0]<ts_gt[-1])
            predictions = predictions_old[idx[0], :]
            
            predictions = predictions_old 
            
            
            # ts_pred = predictions[idx[0], 0]
            ts_pred = predictions[:, 0]
            timestamps.append(ts_pred)
            skeletons_gt = np.zeros((len(ts_pred), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
            skeletons_gt = np.zeros((len(ts_pred), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
            for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
                skeletons_gt[:, k_map[1], 0] = data[k_map[0]]['x'](ts_pred)
                skeletons_gt[:, k_map[1], 1] = data[k_map[0]]['y'](ts_pred)

            skeletons_pred = predictions[:, 2:].reshape(len(predictions), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)
  
            # resample GT at 1 KHz for rmse
            fHz = 1000
            tGT1K = np.arange(ts_gt[0], ts_gt[-1], 1/fHz)
            skeletons_gt1K = np.zeros((len(tGT1K), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
            for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
               skeletons_gt1K[:, k_map[1], 0] = data[k_map[0]]['x'](tGT1K)
               skeletons_gt1K[:, k_map[1], 1] = data[k_map[0]]['y'](tGT1K)
            pred1K = np.zeros([len(tGT1K),28])
            pred1K[:,0] = tGT1K
            for i in range(1,28):
                interp = np.interp(tGT1K, ts_pred, predictions[:,i])
                pred1K[:,i] = interp

            # predictions
            skeletons_pred = predictions[:, 2:].reshape(len(predictions), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)
            skeletons_pred1K = pred1K[:, 2:].reshape(len(pred1K), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)          
            
            # differenciate GT poses
            vel_gt1K_noF = np.gradient(skeletons_gt1K, axis=0)*fHz
            # vx = np.gradient(skeletons_gt1K[:,:,0], axis=1) 
            # vel_pred1K_noF = np.gradient(skeletons_pred1K, axis=0)*fHz
            vel_pred1K_noF = skeletons_pred1K * 1
            
            # filter velocity estimations
            from scipy.signal import savgol_filter
            vel_gt1K = savgol_filter(vel_gt1K_noF, 501, 3, axis=0) 
            vel_pred1K = savgol_filter(vel_pred1K_noF, 501, 3, axis=0) 
            
            # vel_gt1K = vel_gt1K_noF
            # vel_pred1K = vel_pred1K_noF

            algorithm_names.append(algo_name)
            skeletons_predictions.append(skeletons_pred)
            skeletons_predictions1K.append(vel_pred1K)

            # compute PCK
            if len(args.pck) != 0:

                if 'pck' not in results['datasets'][results_key].keys():
                    results['datasets'][results_key]['pck'] = dict()

                for thi, th in enumerate(args.pck):

                    # update dataset metric
                    pck = metrics_utils.PCK(threshold=th)
                    # pck.update_samples(skeletons_pred, skeletons_gt, pck_sizes_gt_interp(ts_pred))
                    pck.update_samples(skeletons_pred1K, skeletons_gt1K, pck_sizes_gt_interp(tGT1K))

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
                rmse.update_samples(vel_pred1K, vel_gt1K)

                algo_name = pred_path.stem
                results['datasets'][results_key]['rmse'][algo_name] = rmse

                # update algo metric
                if 'rmse' not in results['global'].keys():
                    results['global']['rmse'] = dict()

                if algo_name not in results['global']['rmse'].keys():
                    results['global']['rmse'][algo_name] = metrics_utils.RMSE()

                rmse = results['global']['rmse'][algo_name]
                rmse.update_samples(vel_pred1K, vel_gt1K)
                
            # compute latency
            if args.lat:
                if 'latency' not in results['datasets'][results_key].keys():
                        results['datasets'][results_key]['latency'] = dict()
                
                latency.append(predictions[:, 1])
                results['datasets'][results_key]['latency'][algo_name] = np.mean(predictions[:, 1])
                # latency = np.mean(predictions[:, 1])
                # results['datasets'][results_key]['latency']rt[algo_name] = np.arange(1)
                # results['datasets'][results_key]['latency'][algo_name][0] = np.mean(latency)
            
            # compute MPJPE
            if args.mpjpe:

                if 'mpjpe' not in results['datasets'][results_key].keys():
                    results['datasets'][results_key]['mpjpe'] = dict()

                mpjpe = metrics_utils.MPJPE()
                mpjpe.update_samples(vel_pred1K, vel_gt1K)

                algo_name = pred_path.stem
                results['datasets'][results_key]['mpjpe'][algo_name] = mpjpe

                # update algo metric
                if 'mpjpe' not in results['global'].keys():
                    results['global']['mpjpe'] = dict()

                if algo_name not in results['global']['mpjpe'].keys():
                    results['global']['mpjpe'][algo_name] = metrics_utils.MPJPE()

                mpjpe = results['global']['mpjpe'][algo_name]
                mpjpe.update_samples(vel_pred1K, vel_gt1K)
            

        if(args.plot_traj):
            plot_predictions(output_ds_folder_path, results_key, tGT1K, vel_gt1K, algorithm_names, skeletons_predictions1K)
        if(args.plot_sklt):
            plot_predictions_all_joints(output_ds_folder_path, results_key, tGT1K, skeletons_gt1K, algorithm_names, skeletons_predictions1K)
        if(args.lat):  
            plot_latency(output_ds_folder_path, results_key, timestamps, algorithm_names, latency)
        if(args.sklt):
            plot_skeleton(output_ds_folder_path, results_key, tGT1K, skeletons_gt1K, algorithm_names, skeletons_predictions1K, vel_gt1K)

    # iterate over datasets metrics and print results
    for (ds_name, metrics) in results['datasets'].items():
        
        if(not(to_latex)):
            print("\n==================================================")
            print("Dataset: ", ds_name)
            print("==================================================")
        else:
            print("Dataset ", ds_name, " ✓")

        output_ds_folder_path = output_folder_path / ds_name
        output_ds_folder_path.mkdir(parents=True, exist_ok=True)

        for (metric_name, metric_results) in metrics.items():
            if metric_name == 'pck':

                plot_pck_over_thresholds(metric_results, output_ds_folder_path, ds_name)

                # create table
                header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
                header.append('avg PCK')

                for (th, algos) in metric_results.items():
                    if(not(to_latex)):
                        print("\n-= PCK threshold = ", th, " =-")
                    tabulate_metric_over_algorithms(algos, header,
                                                    descr=f'PCK results for dataset {ds_name}, threshold {th}',
                                                    to_latex=to_latex,
                                                    file_path=output_ds_folder_path,
                                                    file_stem=f'pck_{th}_{ds_name}')

            elif metric_name == 'rmse':
                if(not(to_latex)):
                    print("\n-= RMSE =-")
                # header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in [f'{key} x', f'{key} y']]
                # header.append('avg RMSE x')
                # header.append('avg RMSE y')
                header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in [f'{key}']]
                header.append('avg RMSE')

                tabulate_metric_over_algorithms(metric_results, header,
                                                descr=f'RMSE results for dataset {ds_name}',
                                                to_latex=to_latex,
                                                file_path=output_ds_folder_path,
                                                file_stem=f'rmse_{ds_name}')

                plot_boxplot(metric_results,
                             descr=f'RMSE results for dataset {ds_name}',
                             file_path=output_ds_folder_path / f'rmse_{ds_name}.png')
            
                
            if metric_name == 'latency':
                print("\n-= LATENCY =-")
                header = ['avg latency [ms]']
                tabulate_latency_over_algorithms(metric_results, header,
                                            descr=f'Latency results for dataset {ds_name}',
                                            to_latex=to_latex,
                                            file_path=output_ds_folder_path,
                                            file_stem=f'latency_{ds_name}')
            
            if metric_name == 'mpjpe':
                    print("\n-= MPJPE G=-")
                    header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in
                              [f'{key}']]
                    header.append('MPJPE')

                    tabulate_metric_over_algorithms(metric_results, header,
                                                    descr=f'MPJPE results for dataset {ds_name}',
                                                    to_latex=to_latex,
                                                    file_path=output_ds_folder_path,
                                                    file_stem=f'mpjpe_{ds_name}')
                    plot_boxplot(metric_results,
                                  descr=f'MPJPE results for dataset {ds_name}',
                                  file_path=output_ds_folder_path / f'mpjpe_{ds_name}.png')
                
                
                    

    # iterate over global metrics and print results
    for (metric_name, metric_results) in results['global'].items():
        if(not(to_latex)):
            print("\n==================================================")
            print("Global results: ")
            print("==================================================")
        else:
            print("Global results ✓")

        if metric_name == 'pck':

            plot_pck_over_thresholds(metric_results, output_folder_path)

            # create table
            header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
            header.append('avg PCK')

            for (th, algos) in metric_results.items():
                if(not(to_latex)):
                    print("\n-= PCK threshold = ", th, " =-")
                tabulate_metric_over_algorithms(algos, header,
                                                descr=f'Global PCK results for threshold {th}',
                                                to_latex=to_latex,
                                                file_path=output_folder_path,
                                                file_stem=f'pck_{th}')

        elif metric_name == 'rmse':
            if(not(to_latex)):
                    print("\n-= RMSE =-")
            # create table
            # header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in [f'{key} x', f'{key} y']]
            # header.append('avg RMSE x')
            # header.append('avg RMSE y')
            header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in [f'{key}']]
            header.append('avg RMSE')

            tabulate_metric_over_algorithms(metric_results, header,
                                            descr=f'Global RMSE results',
                                            to_latex=to_latex,
                                            file_path=output_folder_path,
                                            file_stem='rmse')

            plot_boxplot(metric_results,
                         descr=f'Global RMSE results',
                         file_path=output_folder_path / f'rmse.png')
            
            plot_error(metric_results, header,
                         descr=f'Global RMSE results',
                         file_path=output_folder_path / f'rmse.png')
            
        elif metric_name == 'mpjpe':
            if (not (to_latex)):
                print("\n-= MPJPE =-")
            header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in
                      [f'{key}']]
            header.append('avg MPJPE')

            tabulate_metric_over_algorithms(metric_results, header,
                                            descr=f'Global MPJPE results',
                                            to_latex=to_latex,
                                            file_path=output_folder_path,
                                            file_stem='mpjpe')

            plot_boxplot(metric_results,
                          descr=f'Global MPJPE results',
                          file_path=output_folder_path / f'mpjpe.png')
            plot_error(metric_results, header,
                         descr=f'Global MPJPE results',
                         file_path=output_folder_path / f'mpjpe.png')
            
    plt.show()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d', '--datasets_path', help='Path to the folders containing data saved in Yarp format', required=True)
    parser.add_argument('-p', '--predictions_path', help='Path to the predictions folder', required=True)
    parser.add_argument('-i', '--images_folder', help='Path to the folder containing the image frames')
    parser.add_argument('-o', '--output_folder', help='Path to the folder where evaluation results will be saved', required=True)
    parser.add_argument('-pck', help='List of thresholds for computing metric PCK; specifies that PCK must be computed', type=float, nargs='+', default=[])
    parser.add_argument('-rmse', help='flag specifying that the metric RMSE must be computed', dest='rmse', action='store_true')
    parser.set_defaults(rmse=False)
    parser.add_argument('-lat', help='flag specifying that the latency must be computed', dest='lat', action='store_true')
    parser.set_defaults(lat=False)
    parser.add_argument('-latex', help='flag specifying that table results should saved to latex files', dest='latex', action='store_true')
    parser.set_defaults(latex=False)
    parser.add_argument('-pt', help='flag specifying to plot trajectories', dest='plot_traj', action='store_true')
    parser.set_defaults(plot_traj=False)
    parser.add_argument('-ps', help='flag specifying to plot full skeleton trajectories in a single figure', dest='plot_sklt', action='store_true')
    parser.set_defaults(plot_sklt=False)
    parser.add_argument('-mpjpe', help='flag specifying that the metric MPJPE must be computed', dest='mpjpe',
                        action='store_true')
    parser.set_defaults(mpjpe=False)
    parser.add_argument('-sklt', help='flag specifying to plot skeleton with velocity', dest='sklt', action='store_true')
    parser.set_defaults(sklt=False)
    

    args, unknown = parser.parse_known_args()
    if(unknown):
        print('\x1b[1;31;20m' + 'Unknown argument/s: ' + ' '.join(unknown) + '\x1b[0m')

    main(args)
