"""
Functions to visualize the various outputs of movenet
Author: Gaurvi Goyal
"""
import os

import cv2
import numpy as np
import json

DHP19_TO_MOVENET_INDICES = np.array([

    [0, 0],  # head
    [1, 2],  # lshoulder
    [2, 1],  # rshoulder
    [3, 4],  # lelbow
    [4, 3],  # relbow
    [5, 8],  # lwrist
    [6, 7],  # rwrist
    [7, 5],  # lhip
    [8, 6],  # rhip
    [9, 10],  # lknee
    [10, 9],  # rknee
    [11, 12],  # lankle
    [12, 11]  # rankle
])
hpecore_kps_labels = {'head': 0,
                      'shoulderR': 1,
                      'shoulderL': 2,
                      'elbowR': 3,
                      'elbowL': 4,
                      'hipL': 5,
                      'hipR': 6,
                      'handR': 7,
                      'handL': 8,
                      'kneeR': 9,
                      'kneeL': 10,
                      'footR': 11,
                      'footL': 12
                      }
lines_in_skeleton = [
    ['shoulderR', 'elbowR'],
    ['elbowR', 'handR'],
    ['shoulderL', 'elbowL'],
    ['elbowL', 'handL'],
    ['shoulderR', 'shoulderL'],
    ['shoulderR', 'hipR'],
    ['shoulderL', 'hipL'],
    ['hipR', 'hipL'],
    ['hipR', 'kneeR'],
    ['kneeR', 'footR'],
    ['hipL', 'kneeL'],
    ['kneeL', 'footL'],
]
upper_joints = ['head', 'shoulderR', 'shoulderL', 'elbowR', 'elbowL', 'handR', 'handL']

MOVENET_TO_DHP19_INDICES = DHP19_TO_MOVENET_INDICES[np.argsort(DHP19_TO_MOVENET_INDICES[:, 1]), :]


def movenet_to_hpecore(pose):
    return pose[MOVENET_TO_DHP19_INDICES[:, 0], :]


def get_kps_names_hpecore(kps):
    kps_out = {}
    for key, value in hpecore_kps_labels.items():
        kps_out[key] = np.array(kps[value, :])
    return kps_out


def superimpose_pose(img_in, pose, num_classes=13, tensors=True, filename=None):
    """ inputs:
            img: rgb image of any size
            pose: numpy array for size (2,:) or 1d array in (x y x y ..) configuration
    """
    print(pose)
    pose = np.array(pose)
    pose = pose.squeeze()
    pose = pose.reshape((num_classes, -1))
    img = np.copy(img_in)
    img = img.astype(np.uint8)
    if tensors:
        img = np.transpose(img.cpu().numpy(), axes=[1, 2, 0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    h, w = img.shape[:2]
    # print("w is ", w)
    # print("h is ", h)
    if np.max(pose) < 1:
        for i in range((pose.shape[0])):
            pose[i, 0] = int(pose[i, 0] * w)
            pose[i, 1] = int(pose[i, 1] * h)
    radius = 3
    thickness = 2
    for i in range((pose.shape[0])):
        # img = cv2.circle(img,(5,5),3,(0,255,0),5)
        if pose.shape[1] == 3:
            radius = int(pose[i, 2]*5)
        img = cv2.circle(img, (int(pose[i, 0]), int(pose[i, 1])), radius, color, thickness)
    if filename is not None:
        # print(filename)
        cv2.imwrite(filename, img)

    cv2.imshow('a', img)
    cv2.waitKey(100)


def add_skeleton(img, keypoints, color, lines=None, normalised=True, th=0.1, confidence=False, upper=False, text=False):
    keypoints = keypoints.reshape([-1])
    if len(img.shape) == 2:
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
    if normalised:
        h, w, _ = img.shape
    else:
        h = w = 1
    if keypoints.size % 3 == 0:
        dim = 3
    else:
        dim = 2
    radius = 2
    use_color = color
    for i in range(len(keypoints) // dim):
        if upper and list(hpecore_kps_labels.keys())[i] not in upper_joints:
            continue
        x = int(keypoints[i * dim] * w)
        y = int(keypoints[i * dim + 1] * h)
        if dim == 3:
            if confidence:
                radius = int(keypoints[i * dim + 2] * 10)
            if keypoints[i * dim + 2] < th:
                use_color = (0, 0, 0)
            else:
                use_color = color
        cv2.circle(img, (x, y), radius, use_color, 2)
        if text:
            img = cv2.putText(img, str(f"{keypoints[i * dim + 2]:.3f}"), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, color, thickness=1, lineType=cv2.LINE_AA)
            # img = cv2.putText(img, list(hpecore_kps_labels.keys())[i], (x,y), cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.5, color, thickness=1, lineType=cv2.LINE_AA)
    if lines is not None:
        kps_2d = np.reshape(keypoints, [-1, dim])
        kps_dict = get_kps_names_hpecore(kps_2d)
        for [jointa, jointb] in lines_in_skeleton:
            if kps_dict[jointa][2] < th or kps_dict[jointb][2] < th:
                continue
            if upper and (jointa not in upper_joints or jointb not in upper_joints):
                continue
            # print('kps_dict[jointa]', kps_dict[jointa][0])
            x1 = int(kps_dict[jointa][0] * w)
            y1 = int(kps_dict[jointa][1] * h)
            x2 = int(kps_dict[jointb][0] * w)
            y2 = int(kps_dict[jointb][1] * h)

            cv2.line(img, (x1, y1), (x2, y2), color, thickness=2)
    return img

# if __name__ == "__main__":
#     file_img = '/home/ggoyal/data/mpii/tos_synthetic_export/000041029.jpg'
#     pose_path = '/home/ggoyal/data/mpii/poses_norm.json'
#
#     img = cv2.imread(file_img)
#     img_basename = os.path.basename(file_img)
#     with open(pose_path, 'r') as f:
#         train_label_list = json.loads(f.readlines()[0])
#     for line in train_label_list:
#         if line["img_name"] == img_basename:
#             pose = line["keypoints"]
#
#     superimpose_pose(img,pose, cfg["num_classes"])
