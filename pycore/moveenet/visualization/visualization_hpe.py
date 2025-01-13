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

def viz_prediction_all_joints(algo_names, skeletons_predictions, skeletons_gt, ts):
     '''
     
     '''
     res = [480, 640]
     image = np.zeros(res, np.uint8)
     thickness = 2
     count = 0
    #  file_path = save_video
     file_path = '/home/cpham-iit.local/data/h36m/videos/gnn_eval.mp4'
     frame_width = 640
     frame_height = 480
     fps = 30
     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
     output = cv2.VideoWriter(file_path, fourcc, fps, (frame_width, frame_height))
     print('saving video')

     for ts_idx in range(len(ts)):
        
        # print('skt type',skeletons_predictions[0][i,:])
        # cv2.rectangle(image, (520,420), (640,480), (255,255,255), int(thickness/2)) #img size [480, 640]
        # cv2.line(image, (530, 440),(550,440), (255,0,0), thickness) #MoveEnet
        # cv2.putText(image, algo_names[0], (570, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), int(thickness/2), cv2.LINE_AA)
        # cv2.line(image, (530, 460),(550,460), (0,255,0), thickness) #OpenPoseRGB
        # cv2.putText(image, algo_names[1], (570, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), int(thickness/2), cv2.LINE_AA)
        image1 = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2BGR)
        image2 = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2BGR)
        image3 = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2BGR)
        image1 = add_skeleton(image1, skeletons_predictions[0][ts_idx,:].flatten(), (255, 0, 0), lines=True, normalised=False)
        image1 = cv2.resize(image1, (res[1], res[0]))
        image2 = add_skeleton(image2,skeletons_predictions[1][ts_idx,:].flatten(), (0,255,0), lines=True, normalised=False)
        image2 = cv2.resize(image2,(res[1], res[0]))
        image3 = add_skeleton(image3,skeletons_gt[ts_idx,:].flatten(), (0,0,255), lines=True, normalised=False)
        image3 = cv2.resize(image3,(res[1], res[0]))
        viz_openpose_moveenet = cv2.addWeighted(image1,1,image2,1,0)
        openpose_moveenet_GT = cv2.addWeighted(viz_openpose_moveenet,1,image3,1,0)
        # cv2.rectangle(openpose_moveenet_GT, (520,420), (640,480), (255,255,255), int(thickness/2)) #img size [480, 640]
        cv2.line(openpose_moveenet_GT, (530, 420),(550,420), (0,0,255), thickness) #GT
        cv2.putText(openpose_moveenet_GT, 'GT', (570, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), int(thickness/2), cv2.LINE_AA)
        cv2.line(openpose_moveenet_GT, (530, 440),(550,440), (255,0,0), thickness) #MoveEnet
        cv2.putText(openpose_moveenet_GT, algo_names[0], (570, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), int(thickness/2), cv2.LINE_AA)
        cv2.line(openpose_moveenet_GT, (530, 460),(550,460), (0,255,0), thickness) #OpenPoseRGB
        cv2.putText(openpose_moveenet_GT, algo_names[1], (570, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), int(thickness/2), cv2.LINE_AA)
        output.write(openpose_moveenet_GT)
        # image = add_skeleton(image, skeletons_predictions[0][ts_idx,:].flatten(), (255, 0, 0), lines=True, normalised=False)
        # # image1 = cv2.resize(image1, (res[1], res[0]))
        # image = add_skeleton(image,skeletons_predictions[1][ts_idx,:].flatten(), (0,255,0), lines=True, normalised=False)
        # # image2 = cv2.resize(image2,(res[1], res[0]))
        # image = add_skeleton(image,skeletons_gt[ts_idx,:].flatten(), (0,0,255), lines=True, normalised=False)
        # # print(skeletons_predictions[0][ts_idx,:].flatten())
        # # print(skeletons_predictions[1][ts_idx,:].flatten())
        # # print(skeletons_gt[ts_idx,:].flatten())
        # # exit()
        # # image3 = cv2.resize(image3,(res[1], res[0]))
        # # dst = cv2.addWeighted(image1,1/3,image2,1/3,image3,1/3,0)
        # image = image.astype('uint8')
        # output.write(image)
     cv2.destroyAllWindows()
     output.release()
     return None

def viz_all_joints(output_folder_path, ds_name, timestamps, joints_gt, algo_names, joints_predicted):
    #iterate the samples
        #iterate the algorithms
        #Plot predicted joints at timestamps freq 1000Hz
        #Plot GT joints at timestamps freq 1000Hz
    #
    pass



def create_pred_GT_pairing(predictions_folder,datasets_path,multi_channel):
    '''
    Make a data samples dictionary and store prediction path and GT path in the dictionary
    '''
    data_samples = {}
    pred_samples = os.listdir(predictions_folder) #get the list of all files and directories in predictions folder
    if multi_channel:
        for subsample in pred_samples:
            sample = '_'.join(subsample.split('_')[1:])
            channel = re.findall('[0-9]+', subsample.split('_')[0])[0]
            for folder in (datasets_path / sample).iterdir():
                if channel in folder.name and 'skeleton' in folder.name:
                    data_samples[subsample] = datasets_path / folder / 'data.log'
                    continue
    else:
        for sample in pred_samples:
            yarp_path_dir = datasets_path / sample
            yarp_path = [x for x in yarp_path_dir.iterdir() if 'skeleton' in x.name][0] / 'data.log'
            data_samples[sample] = yarp_path #adding a new key-value pair
    return data_samples

def main(args):
    plt.close('all')

    output_folder_path = Path(args.output_folder).resolve()
    output_folder_path.mkdir(parents=True, exist_ok=True)

    predictions_folder = Path(args.predictions_path)
    results = dict()
    results['datasets'] = dict()
    results['global'] = dict()

    # import GT from yarp
    datasets_path = Path(args.datasets_path) #path/to/datasets/folder/. Eg. ../EV2/

    data_samples = create_pred_GT_pairing(predictions_folder,datasets_path, multi_channel = False)

    for sample, yarp_path in tqdm(data_samples.items()):
        '''
        Load prediction file and GT. tqdm means in progress, it make your loops show a smart progress meter
        '''
        predictions_path = predictions_folder / sample
        # yarp_path_dir = datasets_path / sample ## This should have the next level folder list as well.

        if yarp_path.exists() == False:
            print('\x1b[1;33;20m' + "Skipping " + str(sample) + " as no GT available at " + str(yarp_path) + '\x1b[0m')
            continue

        numbers = re.findall('[0-9]+', yarp_path.parent.name)
        channel_id = numbers[0]
        parent_folder_prefix = yarp_path.parent.name.split(channel_id)[0]
        channel_folder = f'{parent_folder_prefix}{channel_id}'
        dataset_name = yarp_path.parent.parent.name
        results_key = f'{dataset_name}_{channel_folder}'

        # predictions_path = Path(args.predictions_path) / dataset_name
        predictions_file_path = list(predictions_path.glob('**/*.csv'))
        # predictions_file_path.sort()
        if len(predictions_file_path) == 0:
            print('\x1b[1;33;20m' + "Skipping " + str(dataset_name) + " as no results exist in" + str(
                predictions_path) + '\x1b[0m')
            continue

        predictions_file_path.sort(reverse=True)
        data = ds_parsing.import_yarp_skeleton_data(yarp_path) # parsing GT data into array containing ts, head sizes and torso sizes

        ts_gt = np.concatenate(([.0], data['ts'], [data['ts'][-1] + 1]))

        # interpolate ground truth joints so that they can be compared with the high frequency predictions
        for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
            x_interpolation = interpolate.interp1d(ts_gt, np.concatenate(
                ([data[k_map[0]][0, 0]], data[k_map[0]][:, 0], [data[k_map[0]][-1, 0]])))  # , fill_value="extrapolate"
            y_interpolation = interpolate.interp1d(ts_gt, np.concatenate(
                ([data[k_map[0]][0, 1]], data[k_map[0]][:, 1], [data[k_map[0]][-1, 1]])))
            data[k_map[0]] = dict()
            data[k_map[0]]['x'] = x_interpolation
            data[k_map[0]]['y'] = y_interpolation

        # GT contains the size of the torso
        if data['head_sizes'][0] == -1:
            pck_sizes_gt_interp = interpolate.interp1d(ts_gt, np.concatenate(
                ([data['torso_sizes'][0]], data['torso_sizes'], [data['torso_sizes'][-1]])))
        # GT contains the size of the head
        else:
            pck_sizes_gt_interp = interpolate.interp1d(ts_gt, np.concatenate(
                ([data['head_sizes'][0]], data['head_sizes'], [data['head_sizes'][-1]])))
            
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
            if algo_name in args.exclude: # exclude algos as requested in command line.
                continue
            try:
                predictions_old = np.loadtxt(str(pred_path.resolve()), dtype=float)
            except ValueError:
                with open(str(pred_path.resolve())) as f:
                    content = f.readlines()
                for l, line in enumerate(content):
                    predictions_old[l,:] = np.asarray(line.split(','))
            predictions_old = predictions_old[predictions_old[:, 0].argsort()]
            # ts_pred = predictions[:, 0]
            idx = np.where(np.logical_and(predictions_old[:, 0] > 0.1 * ts_gt[-1], predictions_old[:, 0] < ts_gt[-1])) #cropped record time [0.1 total time, total time]
            # idx = np.where(predictions_old[:, 0]<ts_gt[-1])
            predictions = predictions_old[idx[0], :]
            # ts_pred = predictions[idx[0], 0]
            ts_pred = predictions[:, 0]
            print(f'{algo_name} predicted timestamp: ', ts_pred)
            # print('GT timestamp: ', ts_gt)
            timestamps.append(ts_pred)
            skeletons_gt = np.zeros((len(ts_pred), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
            for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
                skeletons_gt[:, k_map[1], 0] = data[k_map[0]]['x'](ts_pred)
                skeletons_gt[:, k_map[1], 1] = data[k_map[0]]['y'](ts_pred)
            try:
                skeletons_pred = predictions[:, 2:].reshape(len(predictions),
                                                        len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)
            except ValueError:
                print(sample)
                print(pred_path)
                print(f"Looks like prediction file {pred_path} is mostly empty.")
                continue

            fHz = args.frequency
            tGT1K = np.arange(ts_gt[0], ts_gt[-1], 1 / fHz)
            skeletons_gt1K_old = np.zeros((len(tGT1K), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
            for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
                skeletons_gt1K_old[:, k_map[1], 0] = data[k_map[0]]['x'](tGT1K)
                skeletons_gt1K_old[:, k_map[1], 1] = data[k_map[0]]['y'](tGT1K)
            pred1K_old = np.zeros([len(tGT1K), 28])
            pred1K_old[:, 0] = tGT1K
            for i in range(1, 28):
                interp = np.interp(tGT1K, ts_pred, predictions[:, i])
                pred1K_old[:, i] = interp
            print('GT timestamp: ', tGT1K)
            pred1K_old = pred1K_old[pred1K_old[:, 0].argsort()]
            idx1 = np.where(np.logical_and(pred1K_old[:, 0] > 0.1 * tGT1K[-1], pred1K_old[:, 0] < tGT1K[-1]))
            pred1K = pred1K_old[idx1[0], :]
            # ts_pred = predictions[idx[0], 0]
            ts_pred1K = pred1K[:, 0]

            #GT chunk
            # skeletons_gt1K_old = skeletons_gt1K_old[skeletons_gt1K_old[:, 0].argsort()]
            # idx1 = np.where(np.logical_and(pred1K_old[:, 0] > 0.1 * tGT1K[-1], pred1K_old[:, 0] < tGT1K[-1]))
            skeletons_gt1K = skeletons_gt1K_old[idx1[0], :]
            # ts_pred = predictions[idx[0], 0]

            # print(f'{algo_name} ts pred1K freq: ', 1/(ts_pred1K[1] - ts_pred1K[0]))
            # predictions
            # print(ts_gt[-1])
            skeletons_pred = predictions[:, 2:].reshape(len(predictions),
                                                        len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)
            skeletons_pred1K = pred1K[:, 2:].reshape(len(pred1K), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)
            
            
            algorithm_names.append(algo_name)
            skeletons_predictions.append(skeletons_pred)
            skeletons_predictions1K.append(skeletons_pred1K)
        # print(skeletons_gt1K.shape)
        # exit()
        # print(len(skeletons_predictions))
        
    # viz_all_joints(output_folder_path = output_ds_folder_path, ds_name = dataset_name , timestamps = timestamps, joints_gt = skeletons_gt1K, algo_names = algorithm_names,
                    # joints_predicted = skeletons_predictions1K)
    viz_prediction_all_joints(algo_names = algorithm_names, skeletons_predictions=skeletons_predictions1K, skeletons_gt=skeletons_gt1K, ts = ts_pred1K)
    # output_folder_path = Path('/home/cpham-iit.local/data/h36m/videos/')

    # # predictions_folder = Path(args.predictions_path)
    # predictions_path = Path('/home/cpham-iit.local/data/cam2_S9_Photo')

    # predictions_file_path = list(predictions_path.glob('**/*.csv'))
    # print(predictions_file_path)
    # # if len(predictions_file_path) == 0:
    # #     print('\x1b[1;33;20m' + "Skipping " + " as no results exist in" + str(
    # #         predictions_path) + '\x1b[0m')
    # #     continue
    # res = [480, 640]
    # yarp_path = Path(args.datasets_path)
    # data = ds_parsing.import_yarp_skeleton_data(yarp_path)
    # # ground truth in yarp format is supposed to be stored in folders name <dataset_name>/ch<channel_id>[frequency_info]skeleton
    #     # find the channel id
    # numbers = re.findall('[0-9]+', yarp_path.parent.name)
    # channel_id = numbers[0]
    # parent_folder_prefix = yarp_path.parent.name.split(channel_id)[0]
    # channel_folder = f'{parent_folder_prefix}{channel_id}'
    # dataset_name = yarp_path.parent.parent.name
    # results_key = f'{dataset_name}_{channel_folder}'

    # ts_gt = np.concatenate(([.0], data['ts'], [data['ts'][-1] + 1]))

    # predictions_file_path.sort(reverse=True)
    # algorithm_names = []
    # skeletons_predictions = []
    # timestamps = []
    # latency = []
    # #parse prediction
    # for pred_path in predictions_file_path:
    #     print('pred path',pred_path)
    #     algo_name = pred_path.stem 
    #     if algo_name in args.exclude:
    #         continue
    #     try:
    #         predictions_old = np.loadtxt(str(pred_path.resolve()),dtype= float)
    #     except ValueError:
    #         with open(str(pred_path.resolve())) as f:
    #                 content = f.readlines()
    #         for l, line in enumerate(content):
    #              predictions_old[l,:] = np.asarray(line.split(','))

    #     predictions_old = predictions_old[predictions_old[:, 0].argsort()]

    #     idx = np.where(np.logical_and(predictions_old[:, 0] > 5, predictions_old[:, 0] < 30))
    #     predictions = predictions_old[idx[0], :]
    #     print('predictions: ', predictions.shape)
    #     ts_pred = predictions[:,0]
    #     print('time stamps: ', ts_pred.shape)
    #     timestamps.append(ts_pred)

    #     skeletons_gt = np.zeros((len(ts_pred), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
    #     for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
    #         skeletons_gt[:, k_map[1], 0] = data[k_map[0]]['x'](ts_pred)
    #         skeletons_gt[:, k_map[1], 1] = data[k_map[0]]['y'](ts_pred)

    #     skeletons_pred = predictions[:, 2:].reshape(len(predictions),
    #                                                     len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)
    #     algorithm_names.append(algo_name)
    #     skeletons_predictions.append(skeletons_pred)

    #     # print('skt pred',skeletons_pred[0,:].shape)
    # viz_prediction_all_joints(algorithm_names, skeletons_predictions, skeletons_gt)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d', '--datasets_path', help='Path to the folders containing data saved in Yarp format',
                        required=False)
    parser.set_defaults(datasets_path = '/home/cpham-iit.local/data/h36m_full/EV2')
    parser.add_argument('-p', '--predictions_path',
                        help='Path to the predictions folder containing subfolders with results in .csv format.',
                        required=False)
    parser.set_defaults(predictions_path = '/home/cpham-iit.local/data/h36m/samples/test_val')
    parser.add_argument('-o', '--output_folder', help='Path to the folder where evaluation results will be saved',
                        required=False)
    parser.set_defaults(output_folder = '/home/cpham-iit.local/data/visualization')
    parser.add_argument('-lat', help='flag specifying that the latency must be computed', dest='lat',
                        action='store_true')
    parser.set_defaults(lat=False)
    parser.add_argument('-f', '--frequency', help='Evaluation frequency',type=int,default=250)
    parser.add_argument('-e', '--exclude', action='append', default=[],
                        help='Exclude specific algorithms from results. Add a new -e for each algo.', required=False)
    args, unknown = parser.parse_known_args()
    if (unknown):
        print('\x1b[1;31;20m' + 'Unknown argument/s: ' + ' '.join(unknown) + '\x1b[0m')

    main(args)