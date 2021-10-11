# %% Start
import numpy as np
import pickle
from pathlib import Path
from evaluation.utils import metrics
from evaluation.utils import parse_openpose_keypoints_json
from evaluation.dhp19.utils import DHP19_BODY_PARTS, openpose_to_dhp19

# selected recording
subj, sess, mov = 1, 2, 3
recording = 'S{}_{}_{}'.format(subj, sess, mov)
cam = str(3)

# read data
# poses_gt = np.load('/home/fdipietro/hpe-data/2d_Nicolo/' + recording +'/2d_poses_cam_3_7500_events.npy')
# poses_pred_files = sorted(Path('/home/fdipietro/hpe-data/open-pose/' + recording).glob('*.json'))
# image_files = sorted(Path('/home/fdipietro/hpe-data/grayscale/' + recording +'/' + cam +'/reconstruction').glob('*.png'))
# t_op = np.load('/home/fdipietro/hpe-data/2d_Nicolo/' + recording +'/2d_poses_cam_3_7500_ts.npy')

poses_pred = np.load('/home/ggoyal/data/dhp19_sample/outputs/'+recording+'/pred_2D.npy', allow_pickle= True)
t_op = np.load('/home/ggoyal/data/dhp19_sample/outputs/'+recording+'/pred_ts.npy', allow_pickle= True)
gt2Dfile = '/home/ggoyal/data/dhp19_sample/subset/'+recording+'/ground-truth/'+recording+'.pkl'# + recording +'.pkl'
with open(gt2Dfile, 'rb') as f:
    data = pickle.load(f)
    
# build full GT (for plots)
L = len(data['ch3']['ts'])
gt = np.empty([L,13,2])
for j in range(L):
    i=0
    for key, value in data['ch3'].items():
        if(key!='ts'):
            gt[j,i,:] = value[j]
            i=i+1
t_gt = np.array(data['ch3']['ts'])

# Frame rate information
diff = np.ediff1d(t_op)
print('Mean freq = ', 1/np.mean(diff))
print('Min freq = ', 1/np.max(diff))
print('Max freq = ', 1/np.min(diff))
print('\n')

# check if the number of images, gt and predicted poses is the same
# assert len(poses_gt) == len(poses_pred_files)
# assert len(poses_pred_files) == len(image_files)

# read and preprocess predicted poses
# poses_pred = np.zeros((len(poses_pred), len(DHP19_BODY_PARTS), 2), dtype=int)
# for pi, path in enumerate(poses_pred):
#     op_pose = parse_openpose_keypoints_json(path)
#     poses_pred[pi, :] = openpose_to_dhp19(op_pose)


# compute metrics
# poses_gt_flip = np.array(poses_gt) # bug: x and y are fliped for some reason -> should be fixed
# for i in range(len(poses_gt)):
#     poses_gt_flip[i,:,:] = np.fliplr(poses_gt[i,:,:])

# mpjpe = metrics.compute_mpjpe(poses_pred, poses_gt_flip)
# mpjpe = metrics.compute_mpjpe(poses_pred, gt)
# metrics.print_mpjpe(mpjpe, list(DHP19_BODY_PARTS.keys()))

# save metric for bar graph plots
# np.save('/home/fdipietro/hpe-data/op-eval_Franco/m_' +  recording + '.npy', mpjpe, allow_pickle=False)


# %% Plot joint vs t (one joint per figure)
import matplotlib.pyplot as plt

#select interesting joints for each dataset
if(recording == 'S1_1_1'):
    J = [DHP19_BODY_PARTS['head'], DHP19_BODY_PARTS['elbowL'], DHP19_BODY_PARTS['shoulderL']]
elif(recording == 'S1_2_1'):
    J = [DHP19_BODY_PARTS['handL'], DHP19_BODY_PARTS['kneeL'], DHP19_BODY_PARTS['footR']]
elif(recording == 'S1_2_3'):
    J = [DHP19_BODY_PARTS['head'], DHP19_BODY_PARTS['shoulderL'], DHP19_BODY_PARTS['kneeR']]
elif(recording == 'S1_2_4'):
    J = [DHP19_BODY_PARTS['head'], DHP19_BODY_PARTS['shoulderL'], DHP19_BODY_PARTS['footL']]
elif(recording == 'S1_3_3'):
    J = [DHP19_BODY_PARTS['handL'], DHP19_BODY_PARTS['elbowL'], DHP19_BODY_PARTS['shoulderL']]
elif(recording == 'S1_3_6'):
    J = [DHP19_BODY_PARTS['handR'], DHP19_BODY_PARTS['elbowR'], DHP19_BODY_PARTS['shoulderR']]
elif(recording == 'S10_1_4'):
    J = [DHP19_BODY_PARTS['footR'], DHP19_BODY_PARTS['kneeR'], DHP19_BODY_PARTS['hipR']]
elif(recording == 'S13_1_8'):
    J = [DHP19_BODY_PARTS['footR'], DHP19_BODY_PARTS['kneeR'], DHP19_BODY_PARTS['hipR']]
else:
    print('Recording not present')

plt.close('all')
print(poses_pred)
my_dpi = 96
H = 280 #Height of DVS frame
for i in range(len(J)):
    fig = plt.figure(figsize=(2048/my_dpi, 600/my_dpi), dpi=my_dpi)
    ax = plt.subplot(111)
    coord = 1
    ax.plot(t_gt,gt[:,J[i],1-coord], label ='GT [x]')
    ax.plot(t_gt,gt[:,J[i],coord], label ='GT [y]')
    ax.plot(t_op, poses_pred[:,J[i],1-coord], marker = ".", markersize=12, linestyle = 'None', label ='GL-HPE [x]')
    # ax.plot(t_op, poses_gt[:,J[i],1-coord], marker = ".", markersize=12, linestyle = 'None', label ='OP [x]')
    coord=1
    ax.plot(t_op, H-poses_pred[:,J[i],coord], marker = ".", markersize=12, linestyle = 'None', label ='GL-HPE [y]')
    # ax.plot(t_op, poses_gt[:,J[i],coord], marker = ".", markersize=12, linestyle = 'None', label ='OP [y]')
    plt.xlabel('time [sec]', fontsize=14, labelpad=5)
    plt.ylabel('x/y coordinate [px]', fontsize=14, labelpad=5)
    ax.legend(fontsize=12, loc = 'lower right')
    fig.suptitle('Recording: ' + recording + ' - Joint: ' + list(DHP19_BODY_PARTS.keys())[list(DHP19_BODY_PARTS.values()).index(J[i])], fontsize=18, y=0.92)
    # plt.show()
    plt.savefig('/home/ggoyal/data/dhp19_sample/outputs/'+recording+ '_' + list(DHP19_BODY_PARTS.keys())[list(DHP19_BODY_PARTS.values()).index(J[i])] + '.png', bbox_inches='tight')


# # %% BAR GRAPH FULL
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.close('all')
#
# m111 = np.load('/home/fdipietro/hpe-data/op-eval_Franco/m_S1_1_1.npy')
# m121 = np.load('/home/fdipietro/hpe-data/op-eval_Franco/m_S1_2_1.npy')
# m123 = np.load('/home/fdipietro/hpe-data/op-eval_Franco/m_S1_2_3.npy')
# m124 = np.load('/home/fdipietro/hpe-data/op-eval_Franco/m_S1_2_4.npy')
# m133 = np.load('/home/fdipietro/hpe-data/op-eval_Franco/m_S1_3_6.npy')
# m136 = np.load('/home/fdipietro/hpe-data/op-eval_Franco/m_S1_3_6.npy')
# m1014 = np.load('/home/fdipietro/hpe-data/op-eval_Franco/m_S10_1_4.npy')
# m1318 = np.load('/home/fdipietro/hpe-data/op-eval_Franco/m_S13_1_8.npy')
#
# width =0.1
# my_dpi=96
# fig = plt.figure(figsize=(2048/my_dpi, 1200/my_dpi), dpi=my_dpi)
#
# plt.grid(axis='y')
# plt.bar(np.arange(len(m111)), m111, width=width, label = 'S1_1_1')
# plt.bar(np.arange(len(m121))+ width, m121, width=width, label = 'S1_2_1')
# plt.bar(np.arange(len(m123))+ 2*width, m123, width=width, label = 'S1_2_3')
# plt.bar(np.arange(len(m124))+ 3*width, m124, width=width, label = 'S1_2_4')
# plt.bar(np.arange(len(m133))+ 4*width, m133, width=width, label = 'S1_3_3')
# plt.bar(np.arange(len(m136))+ 5*width, m136, width=width, label = 'S1_3_6')
# plt.bar(np.arange(len(m1014))+ 6*width, m1014, width=width, label = 'S10_1_4')
# plt.bar(np.arange(len(m1318))+ 7*width, m1318, width=width, label = 'S13_1_8')
# # plt.yscale("log")
# locs, labels = plt.xticks()
# plt.xticks(np.arange(0.35, 13.35, step=1))
# plt.ylabel('Error [px]', fontsize=16, labelpad=5)
# ax = plt.gca()
# ax.tick_params(axis='x', labelrotation = 45)
# ax.legend(fontsize=18, loc = 'upper left', mode = "expand", ncol = 8)
# ax.set_xticklabels(DHP19_BODY_PARTS.keys(), fontsize = 14)
# sns.despine(bottom=True)
# plt.suptitle('Mean Per Joint Position Error', fontsize=18, y=0.92)
# plt.show()
# # plt.savefig('/home/fdipietro/hpe-data/op-eval_Franco/bars.png', bbox_inches='tight')