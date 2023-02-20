import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from scipy import interpolate
from datasets.utils import constants as ds_constants, parsing as ds_parsing

        
        
def avg_vel(ds_name, timestamps, joints_gt):
    
    assert 1 <= joints_gt.shape[2] <= 3, 'coordinates must be either 2D or 3D'
    # iterate on each joint
    data = np.zeros((len(timestamps), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP) * 2))
    keys = []
    for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
        data[:,joint_ind*2] = joints_gt[:, joint_ind, 0]
        data[:,joint_ind*2+1] = joints_gt[:, joint_ind, 1]
        keys.append(joint_key + ' x')
        keys.append(joint_key + ' y')

    df = pd.DataFrame(abs(data), columns=keys)
    # print(df.describe())
    avg75 = np.mean(df.describe().loc['75%'])
    # print("\n75% = " + str(avg75), flush=True)
    avgMax = np.mean(df.describe().loc['max'])
    # print("Max = " + str(avgMax), flush=True)
    avgMean = np.mean(df.describe().loc['mean'])
    # print("Mean = " + str(avgMean), flush=True)
    avgStd = np.mean(df.describe().loc['std'])
    # print("Std = " + str(avgStd), flush=True)    
    
    if(avgMax >= 250 and avgMean >=57):
        print("FAST")
    elif(avgMax < 100 and avgMean < 18):
        print("SLOW")
   
    return avg75, avgMax, avgMean, avgStd
            

def main(args):
    
    plt.close('all')

    # output_folder_path = Path(args.output_folder)
    # output_folder_path = output_folder_path.resolve()
    # output_folder_path.mkdir(parents=True, exist_ok=True)
    results = dict()
    results['datasets'] = dict()
    results['global'] = dict()
    list75 = []
    listMax = []
    listMean = []
    listStd = []
    
    # import GT from yarp
    datasets_path = Path(args.datasets_path)
    yarp_file_paths = list(datasets_path.glob('**/data.log'))

    for yarp_path in tqdm(yarp_file_paths, desc ="Progress: "):
    # for yarp_path in track(yarp_file_paths):

        if 'skeleton' not in yarp_path.parent.name:
            continue
        dataset_name = yarp_path.parent.parent.name
        print('\x1b[1;33;20m' + "Procesing " + str(dataset_name)  + '\x1b[0m')
        data = ds_parsing.import_yarp_skeleton_data(yarp_path)
        ts_gt = np.concatenate(([.0], data['ts'], [data['ts'][-1] + 1]))
    
        # interpolate ground truth joints so that they can be compared with the high frequency predictions
        for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
            x_interpolation = interpolate.interp1d(ts_gt, np.concatenate(([data[k_map[0]][0, 0]], data[k_map[0]][:, 0], [data[k_map[0]][-1, 0]])))
            y_interpolation = interpolate.interp1d(ts_gt, np.concatenate(([data[k_map[0]][0, 1]], data[k_map[0]][:, 1], [data[k_map[0]][-1, 1]])))
            data[k_map[0]] = dict()
            data[k_map[0]]['x'] = x_interpolation
            data[k_map[0]]['y'] = y_interpolation
        skeletons_gt = np.zeros((len(ts_gt), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
        for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
            skeletons_gt[:, k_map[1], 0] = data[k_map[0]]['x'](ts_gt)
            skeletons_gt[:, k_map[1], 1] = data[k_map[0]]['y'](ts_gt)
  
        # resample GT at 1 KHz
        fHz = 1000
        tGT1K = np.arange(ts_gt[0], ts_gt[-1], 1/fHz)
        skeletons_gt1K = np.zeros((len(tGT1K), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
        for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
            skeletons_gt1K[:, k_map[1], 0] = data[k_map[0]]['x'](tGT1K)
            skeletons_gt1K[:, k_map[1], 1] = data[k_map[0]]['y'](tGT1K)
        # differenciate GT poses
        vel_gt1K_noF = np.gradient(skeletons_gt1K, axis=0)*fHz
        # filter velocity
        from scipy.signal import savgol_filter
        vel_gt1K = savgol_filter(vel_gt1K_noF, 901, 3, axis=0) 
        
        # calculate statistics for each dataset
        value75, valueMax, valueMean, valueStd = avg_vel(str(dataset_name), tGT1K, vel_gt1K)
        # add to global list
        list75.append(value75)
        listMax.append(valueMax)
        listMean.append(valueMean)
        listStd.append(valueStd)
    
    dataList = list(zip(list75, listMax, listMean, listStd))
    df = pd.DataFrame(dataList, columns=['75%', 'Max', 'Mean', 'Std'])
    # print(df.describe())
    
    # define clusters for mean velocity
    df['mean_group'] = pd.cut(df['Mean'], bins=range(5, 71, 13))
    # define clusters for max velocity
    df['max_group'] = pd.cut(df['Max'], bins=range(50, 301, 50))
    
    # plots
    my_dpi = 96
    fig1 = plt.figure(figsize=(2048/my_dpi, 900/my_dpi), dpi=my_dpi)
    ax1 = sns.boxplot(x="mean_group", y="Mean", data=df)
    ax1.set_xlabel("Speed [px/s]", fontsize=24)
    ax1.set_ylabel("Speed [px/s]", fontsize=24)
    fig1.suptitle('Mean velocity distribution', fontsize=32, y=0.92)
    plt.xticks(fontsize=22, rotation=0)
    
    fig2 = plt.figure(figsize=(2048/my_dpi, 900/my_dpi), dpi=my_dpi)
    ax2 = sns.boxplot(x="max_group", y="Max", data=df)
    ax2.set_xlabel("Speed [px/s]", fontsize=24)
    ax2.set_ylabel("Speed [px/s]", fontsize=24)
    fig2.suptitle('Max velocity distribution', fontsize=32, y=0.92)
    plt.xticks(fontsize=22, rotation=0)
    
    # print statistics
    print(df)
    print("Mean values range:")
    print("[", np.min(df['Mean']), ", ", np.max(df['Mean']), "]")
    print("Max values range:")
    print("[", np.min(df['Max']), ", ", np.max(df['Max']), "]")
   
    plt.show()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d', '--datasets_path', help='Path to the folders containing data saved in Yarp format', required=True)
    parser.add_argument('-o', '--output_folder', help='Path to the folder where velocity study results will be saved', required=False)

    args, unknown = parser.parse_known_args()
    if(unknown):
        print('\x1b[1;31;20m' + 'Unknown argument/s: ' + ' '.join(unknown) + '\x1b[0m')

    main(args)