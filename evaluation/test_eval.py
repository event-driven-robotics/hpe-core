#%% BAR GRAPH FULL
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
import pandas as pd
from datasets.utils import constants as ds_constants
plt.close('all')

# Load global PCK@04 results
df_PCK = pd.read_csv("/home/cpham-iit.local/data/ledge10_eval/pck_0.4.txt", sep='\s+')
df_PCK_movenet_cam_24 = df_PCK["movenet_cam-24"].iloc[1:14].values
df_PCK_openpose_rgb = df_PCK["openpose_rgb"].iloc[1:14].values
# df_PCK_hpe_gnn_splineconv_gamer = df_PCK["openpose_rgb"].iloc[1:14].values
#Plot 
width = 0.1
my_dpi = 96
fig = plt.figure(figsize = (2048/my_dpi, 1200/my_dpi), dpi = my_dpi)

plt.grid(axis = 'y')
# plt.bar(np.arange(len()), , width = width, label = 'hpe_gnn_spline_conv_gamer') #S_1_1 = hpe_gnn_spline_conv_gamer 
# plt.bar(np.arange(len(m111)) + width, m111, width = width, label = 'hpe_gnn_spline_conv_ledge')
plt.bar(np.arange(len(df_PCK_movenet_cam_24)) + 2*width, df_PCK_movenet_cam_24, width = width, label = 'MoveeNet')
plt.bar(np.arange(len(df_PCK_openpose_rgb)) + 3*width, df_PCK_openpose_rgb, width = width, label = 'OpenPose_RGB')

locs, labels = plt.xticks()
plt.xticks(np.arange(0.35, 13.35, step = 1))
plt.ylabel('PCK@0.4 [%]', fontsize=16, labelpad = 5)
ax = plt.gca()
ax.tick_params(axis='x', labelrotation = 45)
ax.legend(fontsize = 18, loc='upper left', mode = "expand", ncol = 4)
ax.set_xticklabels(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys(), fontsize = 14)
sns.despine(bottom=True)
plt.suptitle('Global PCK@0.4', fontsize = 18, y = 0.92)
plt.show()
# plt.savefig('', bbox_inches='tight')



