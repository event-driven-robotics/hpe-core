#%% BAR GRAPH FULL
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
import pandas as pd
from datasets.utils import constants as ds_constants
plt.close('all')
PCK_metric = True
MPJPE_metric = False
if PCK_metric:
    # Load global PCK@04 results
    df_PCK = pd.read_csv("/home/cpham-iit.local/data/ledge10_eval/pck_0.4.txt", sep='\s+')
    df_PCK_movenet_cam_24 = df_PCK["movenet_cam-24"].iloc[1:14].values
    df_PCK_movenet_cam_24 = [float(x) for x in df_PCK_movenet_cam_24]
    df_PCK_openpose_rgb = df_PCK["openpose_rgb"].iloc[1:14].values
    df_PCK_openpose_rgb = [float(x) for x in df_PCK_openpose_rgb]
    df_PCK_ledge10_solo_weight_contrib_stepwise = df_PCK["ledge10_solo_weight_contrib_stepwise"].iloc[1:14].values
    df_PCK_ledge10_solo_weight_contrib_stepwise = [float(x) for x in df_PCK_ledge10_solo_weight_contrib_stepwise]
    print(type(df_PCK_ledge10_solo_weight_contrib_stepwise[0]))
if MPJPE_metric:
    df_MPJPE = pd.read_csv("/home/cpham-iit.local/data/ledge10_eval/mpjpe.txt", sep='\s+')
    df_MPJPE_movenet_cam_24 = df_MPJPE["movenet_cam-24"].iloc[1:14].values
    df_MPJPE_movenet_cam_24 = [float(x) for x in df_MPJPE_movenet_cam_24]
    df_MPJPE_openpose_rgb = df_MPJPE["openpose_rgb"].iloc[1:14].values
    df_MPJPE_openpose_rgb = [float(x) for x in df_MPJPE_openpose_rgb]
    df_MPJPE_ledge10_solo_weight_contrib_stepwise = df_MPJPE["ledge10_solo_weight_contrib_stepwise"].iloc[1:14].values
    df_MPJPE_ledge10_solo_weight_contrib_stepwise = [float(x) for x in df_MPJPE_ledge10_solo_weight_contrib_stepwise]
#Plot 
width = 0.1
my_dpi = 96
fig = plt.figure(figsize = (2048/my_dpi, 1200/my_dpi), dpi = my_dpi)
plt.grid(axis = 'y')
plt.bar(np.arange(len(df_PCK_ledge10_solo_weight_contrib_stepwise)),df_PCK_ledge10_solo_weight_contrib_stepwise , width = width, label = 'ledge10_solo_weight_contrib_stepwise') #S_1_1 = hpe_gnn_spline_conv_gamer 
plt.bar(np.arange(len(df_PCK_movenet_cam_24)) + width, df_PCK_movenet_cam_24, width = width, label = 'MoveeNet')
plt.bar(np.arange(len(df_PCK_openpose_rgb)) + 2*width, df_PCK_openpose_rgb, width = width, label = 'OpenPose_RGB')

locs, labels = plt.xticks()
plt.xticks(np.arange(0, 13, step = 1))
plt.ylabel('PCK@0.4 [%]', fontsize=16, labelpad = 5)
# plt.ylabel('MPJPE [px]', fontsize=16, labelpad = 5)
ax = plt.gca()
ax.tick_params(axis='x', labelrotation = 30)
ax.legend(fontsize = 18, loc='upper left', mode = "expand", ncol = 5)
ax.set_xticklabels(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys(), fontsize = 14)
ax.set_ylim(0, 1)
# sns.despine(bottom=True)
plt.suptitle('Global PCK@0.4 for 8 samples', fontsize = 18, y = 0.92)
# plt.suptitle('Global MPJPE for 8 samples', fontsize = 18, y = 0.92)
plt.show()
# plt.savefig('', bbox_inches='tight')



