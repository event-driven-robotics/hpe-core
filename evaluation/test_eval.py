#%% BAR GRAPH FULL
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
import pandas as pd
from datasets.utils import constants as ds_constants
import argparse
plt.close('all')

# Trainer arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pck_path', help='read PCK PATH provided.', default="/home/cpham-iit.local/data/output_hpe_20per/pck_0.4.txt", required=False,
                    type=str)
parser.add_argument('--mpjpe_path', help='read MPJPE PATH provided.', default="/home/cpham-iit.local/data/output_hpe_20per/mpjpe.txt", required=False,
                    type=str)
parser.add_argument("--pck",
                    help="select, to manually select the label for each point",
                    default=True,
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--mpjpe",
                    help="select, to manually select the label for each point",
                    default=False,
                    action=argparse.BooleanOptionalAction)

args = parser.parse_args()

PCK_metric = args.pck
MPJPE_metric = args.mpjpe

if PCK_metric:
    # Load global PCK@04 results
    df_PCK = pd.read_csv(args.pck_path, sep='\s+')
    # df_PCK = pd.read_csv("/home/cpham-iit.local/data/output_hpe_20per/pck_0.4.txt", sep='\s+')
    df_PCK_movenet_cam_24 = df_PCK["movenet_cam-24"]
    df_PCK_movenet_cam_24 = df_PCK_movenet_cam_24.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    # print(df_PCK_movenet_cam_24)
    df_PCK_movenet_cam_24 = df_PCK["movenet_cam-24"].iloc[1:14].values
    df_PCK_movenet_cam_24 = [float(x) for x in df_PCK_movenet_cam_24]
    df_PCK_openpose_rgb = df_PCK["openpose_rgb"]
    df_PCK_openpose_rgb = df_PCK_openpose_rgb.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_PCK_openpose_rgb = df_PCK["openpose_rgb"].iloc[1:14].values
    df_PCK_openpose_rgb = [float(x) for x in df_PCK_openpose_rgb]

    df_PCK_ledge10_solo_weight_contrib_stepwise = df_PCK["ledge10_solo_weight_contrib_stepwise"]
    df_PCK_ledge10_solo_weight_contrib_stepwise = df_PCK_ledge10_solo_weight_contrib_stepwise.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    print(df_PCK_ledge10_solo_weight_contrib_stepwise)
    df_PCK_ledge10_solo_weight_contrib_stepwise = df_PCK["ledge10_solo_weight_contrib_stepwise"].iloc[1:14].values
    df_PCK_ledge10_solo_weight_contrib_stepwise = [float(x) for x in df_PCK_ledge10_solo_weight_contrib_stepwise]

    df_PCK_ledgefull_stepwise_unflipped = df_PCK["ledge_test_stepwise_unflipped"]
    df_PCK_ledgefull_stepwise_unflipped = df_PCK_ledgefull_stepwise_unflipped.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_PCK_ledgefull_stepwise_unflipped = df_PCK["ledge_test_stepwise_unflipped"].iloc[1:14].values
    df_PCK_ledgefull_stepwise_unflipped = [float(x) for x in df_PCK_ledgefull_stepwise_unflipped]
    
if MPJPE_metric:
    df_MPJPE = pd.read_csv(args.mpjpe_path, sep='\s+')
    # df_MPJPE = pd.read_csv("/home/cpham-iit.local/data/output_hpe_20per/mpjpe.txt", sep='\s+')
    df_MPJPE_movenet_cam_24 = df_MPJPE["movenet_cam-24"]
    df_MPJPE_movenet_cam_24 = df_MPJPE_movenet_cam_24.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    # print(df_PCK_movenet_cam_24)
    df_MPJPE_movenet_cam_24 = df_MPJPE["movenet_cam-24"].iloc[1:14].values
    df_MPJPE_movenet_cam_24 = [float(x) for x in df_MPJPE_movenet_cam_24]
    df_MPJPE_openpose_rgb = df_MPJPE["openpose_rgb"]
    df_MPJPE_openpose_rgb = df_MPJPE_openpose_rgb.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_MPJPE_openpose_rgb = df_MPJPE["openpose_rgb"].iloc[1:14].values
    df_MPJPE_openpose_rgb = [float(x) for x in df_MPJPE_openpose_rgb]

    df_MPJPE_ledge10_solo_weight_contrib_stepwise = df_MPJPE["ledge10_solo_weight_contrib_stepwise"]
    df_MPJPE_ledge10_solo_weight_contrib_stepwise = df_MPJPE_ledge10_solo_weight_contrib_stepwise.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    print(df_MPJPE_ledge10_solo_weight_contrib_stepwise)
    df_MPJPE_ledge10_solo_weight_contrib_stepwise = df_MPJPE["ledge10_solo_weight_contrib_stepwise"].iloc[1:14].values
    df_MPJPE_ledge10_solo_weight_contrib_stepwise = [float(x) for x in df_MPJPE_ledge10_solo_weight_contrib_stepwise]

    df_MPJPE_ledgefull_stepwise_unflipped = df_MPJPE["ledge_test_stepwise_unflipped"]
    df_MPJPE_ledgefull_stepwise_unflipped = df_MPJPE_ledgefull_stepwise_unflipped.reindex([0,1,2,3,6,7,4,5,8,9,10,11,12,13])
    df_MPJPE_ledgefull_stepwise_unflipped = df_MPJPE["ledge_test_stepwise_unflipped"].iloc[1:14].values
    df_MPJPE_ledgefull_stepwise_unflipped = [float(x) for x in df_MPJPE_ledgefull_stepwise_unflipped]
#Plot 
width = 0.1
my_dpi = 96
fig = plt.figure(figsize = (2048/my_dpi, 1200/my_dpi), dpi = my_dpi)
plt.grid(axis = 'y')
if PCK_metric:

    plt.bar(np.arange(len(df_PCK_ledge10_solo_weight_contrib_stepwise)),df_PCK_ledge10_solo_weight_contrib_stepwise , width = width, label = 'ledge10_single_weight') #S_1_1 = hpe_gnn_spline_conv_gamer
    plt.bar(np.arange(len(df_PCK_ledgefull_stepwise_unflipped)) + width,df_PCK_ledgefull_stepwise_unflipped , width = width, label = 'ledge_stable') 
    plt.bar(np.arange(len(df_PCK_movenet_cam_24)) + 2*width, df_PCK_movenet_cam_24, width = width, label = 'MoveeNet')
    plt.bar(np.arange(len(df_PCK_openpose_rgb)) + 3*width, df_PCK_openpose_rgb, width = width, label = 'OpenPose_RGB')

    locs, labels = plt.xticks()
    plt.xticks(np.arange(0, 13, step = 1))
    plt.ylabel('PCK@0.4 [%]', fontsize=16, labelpad = 5)
    # plt.ylabel('MPJPE [px]', fontsize=16, labelpad = 5)
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation = 30)
    ax.legend(fontsize = 18, loc='upper left', mode = "expand", ncol = 4, bbox_to_anchor = (0, 1.01, 1, 0.075))
    ax.set_xticklabels(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys(), fontsize = 14)
    ax.set_ylim(0, 1)
    sns.despine(bottom=True)
    plt.suptitle('Global PCK@0.4 for 20per validation set', fontsize = 18, y = 0.98)
    # plt.suptitle('Global MPJPE for 20per validation set', fontsize = 18, y = 0.92)
    plt.show()
    plt.savefig('/home/cpham-iit.local/data/output_hpe_20per/pck.jpeg', bbox_inches='tight')

if MPJPE_metric:

    plt.bar(np.arange(len(df_MPJPE_ledge10_solo_weight_contrib_stepwise)),df_MPJPE_ledge10_solo_weight_contrib_stepwise , width = width, label = 'ledge10_single_weight') #S_1_1 = hpe_gnn_spline_conv_gamer
    plt.bar(np.arange(len(df_MPJPE_ledgefull_stepwise_unflipped)) + width,df_MPJPE_ledgefull_stepwise_unflipped , width = width, label = 'ledge_stable') 
    plt.bar(np.arange(len(df_MPJPE_movenet_cam_24)) + 2*width, df_MPJPE_movenet_cam_24, width = width, label = 'MoveeNet')
    plt.bar(np.arange(len(df_MPJPE_openpose_rgb)) + 3*width, df_MPJPE_openpose_rgb, width = width, label = 'OpenPose_RGB')

    locs, labels = plt.xticks()
    plt.xticks(np.arange(0, 13, step = 1))
    # plt.ylabel('PCK@0.4 [%]', fontsize=16, labelpad = 5)
    plt.ylabel('MPJPE [px]', fontsize=16, labelpad = 5)
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation = 30)
    ax.legend(fontsize = 18, loc='upper left', mode = "expand", ncol = 4, bbox_to_anchor = (0, 1.01, 1, 0.075))
    ax.set_xticklabels(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys(), fontsize = 14)
    ax.set_ylim(0, 50)
    sns.despine(bottom=True)
    # plt.suptitle('Global PCK@0.4 for 20per validation set', fontsize = 18, y = 0.92)
    plt.suptitle('Global MPJPE for 20per validation set', fontsize = 18, y = 0.98)
    plt.show()
    plt.savefig('/home/cpham-iit.local/data/output_hpe_20per/MPJPE.jpeg', bbox_inches='tight')
