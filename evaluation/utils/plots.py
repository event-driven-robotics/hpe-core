import cv2
import numpy as np

from .parsing import JOINT_NOT_PRESENT

ALGO_COLORS = {
    # 'openpose_rgb': 'blue',
    # 'openpose_pim': 'green',
    # 'movenet_cam2': 'orange',
    # 'movenet_cam-24': 'red',
    'movenet_eF': (122.0/255,1.0/255,119.0/255),
    'movenet_wo_finetune': (251.0/255,180.0/255,185.0/255),
    'gl-hpe':(37.0/255,52.0/255,148.0/255),
    'movenet': (197.0/255, 27.0/255, 138.0/255),
    'movenet_wo_pretrain': (247.0/255, 104.0/255, 161.0/255),
    'openpose_rgb': (44.0/255,127.0/255,184.0/255),
    'openpose_pim': (194.0/255,230.0/255,153.0/255),
    'movenet_cam2': (255.0/255, 255.0/255, 255.0/255),
    'movenet_cam-24': (197.0/255, 27.0/255, 138.0/255),
    'movenet_rgb':(37.0/255,52.0/255,148.0/255)
}

OTHER_COLORS = ['black', 'violet','grey', 'pink', 'blue','red','green','yellow', 'cyan', 'orange', 'navy']

RENAMING = {'movenet_cam-24': 'moveEnet',
            'movenet_eF': 'movenet_fixedCount',
            'movenet': 'moveEnet',
            'gl-hpe':'liftmono-hpe',
            'ledge_test_stepwise_unflipped': 'ledge_stable',
            'ledge10_solo_weight_contrib_stepwise': 'ledge10_single_weight',
            'hpe-gnn_two_weight_cone_only_target_connectivity_15':'GraphEnet',
            'hpe-gnn_single_weights_onlytargetloss_connectivity_15_layer_cone_extend': 'GraphEnet_single_weight',

            'hpe-gnn_bothloss_connectivity_0_layer_8': 'connectivity_0',
            'hpe-gnn_bothnodeloss_connectivity_15_layer_8': 'connectivity_15_both_node_loss',
            'hpe-gnn_bothloss_connectivity_20_layer_8' : 'connectivity_20', 
            
            'hpe-gnn_bothloss_connectivity_15_layer_6': '6_layers',
            'hpe-gnn_bothloss_connectivity_15_layer_10': '10_layers',
            'hpe-gnn_nonodeloss_connectivity_15_layer_8': '8_layers_no_loss_loss',
            'hpe-gnn_onlytargetloss_connectivity_15_layer_cone': 'cone_layer'
             

            }


def plot_poses(img, pose_gt, pose_pred):
    for joint_gt, joint_pred in zip(pose_gt, pose_pred):
        cv2.circle(img, (joint_gt[1], joint_gt[0]), 5, (0, 255, 0), cv2.FILLED)
        if _is_no_joint(joint_pred):
            continue
        cv2.circle(img, (joint_pred[0], joint_pred[1]), 5, (0, 0, 255), cv2.FILLED)
        cv2.imshow('',img)
        cv2.waitKey(1)


###########
# PRIVATE #
###########

def _is_no_joint(joint):
    return np.sum(joint == JOINT_NOT_PRESENT) == 2
