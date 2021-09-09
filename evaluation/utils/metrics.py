
import numpy as np


# TODO: what if a gt joint is not present in a frame?
def compute_pck(predicted_joints, gt_joints, threshold, head_size=None):
    """
    Computes PCK (Percentage of Correct Keypoints)
    """
    # compute euclidean distances between joints
    distances = np.linalg.norm(predicted_joints - gt_joints, axis=2)

    # normalize distances according to head size
    if head_size:
        distances = distances / head_size

    # compute correct keypoints
    correct_keypoints = (distances <= threshold).astype(int)

    # compute pck
    pck = np.sum(correct_keypoints, axis=0) / correct_keypoints.shape[0]

    return pck


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
