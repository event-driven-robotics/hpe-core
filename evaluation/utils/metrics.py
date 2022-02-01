
import numpy as np


def compute_pckh(predicted_joints, gt_joints, threshold, head_sizes):
    """
    Computes PCK (Percentage of Correct Keypoints)
    It expects predicted_joints and gt_joints with shape [batch_size, joints_num, 2 or 3] and
    not annotated joints as [-1, -1] or [-1, -1, -1]
    """

    # compute PCK's threshold as percentage of head size in pixels for each pose
    thresholds_head = head_sizes * threshold
    thresholds_head = thresholds_head.reshape([-1, 1]).tile((1, gt_joints.shape[1]))

    # compute euclidean distances between joints
    distances = np.linalg.norm(predicted_joints - gt_joints, axis=2)

    # compute correct keypoints
    correct_keypoints = (distances <= np.array(thresholds_head)).astype(int)

    # remove not annotated keypoints from pck computation
    correct_keypoints = correct_keypoints * (gt_joints[:, :, 0] != -1).astype(int)
    annotated_keypoints_num = np.sum((gt_joints[:, :, 0] != -1).astype(int), axis=0)

    # compute pck
    pck_joints = np.sum(correct_keypoints, axis=0) / annotated_keypoints_num
    pck_avg = np.mean(pck_joints)

    return pck_joints, pck_avg


def compute_mpjpe(predicted_joints, gt_joints):
    """
    Computes MPJPE (Mean Per Joint Position Error)
    """

    # # compute euclidean distances between joints
    distances = np.linalg.norm(predicted_joints - gt_joints, axis=2)
    
    # TODO: what if a gt joint is not present in a frame? get the number of gt joints by
    # gt_num = np.sum(gt_joints == 0)
    mpje = np.sum(distances, axis=0) / len(distances)
    
    
    # compute errors wihtout considering non detected joints    
    dist = np.zeros((len(gt_joints),13))
    count = np.zeros((13))
    for i in range(13):
        for j in range(len(gt_joints)):
            if(predicted_joints[j,i,0]!=0 and predicted_joints[j,i,1]!=0):
                dist[j,i] = np.linalg.norm(predicted_joints[j,i,:] - gt_joints[j,i,:], axis=0)
                count[i] = count[i] + 1
    res = np.sum(dist, axis=0) / count

    # print('distances = ')
    # print(distances[1:4,:])
    # print('dist = ')
    # print(dist[1:4,:])

    # return mpje
    return res

def print_mpjpe(mpjpe, keypoints_str):
    for ei, error in enumerate(mpjpe):
        print(f'{keypoints_str[ei]}: {error}')
