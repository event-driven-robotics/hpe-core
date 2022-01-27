
import argparse
import cv2
import json
import numpy as np
import os
import pathlib

from tqdm import tqdm

import datasets.utils.mat_files as mat_utils
import datasets.mpii.utils.parsing as mpii_parse

from datasets.mpii.utils.parsing import MPII_BODY_PARTS
from datasets.utils.events_representation import EROSSynthetic


MOVENET_KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                    'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                    'right_ankle']

MOVENET_13_TO_MPII_INDICES = np.array([

    [0, None],  # head center, not present in movenet and mpii
    [1, MPII_BODY_PARTS['lshoulder']],  # left shoulder
    [2, MPII_BODY_PARTS['rshoulder']],  # right shoulder
    [3, MPII_BODY_PARTS['lelbow']],  # left elbow
    [4, MPII_BODY_PARTS['relbow']],  # right elbow
    [5, MPII_BODY_PARTS['lwrist']],  #  left wrist
    [6, MPII_BODY_PARTS['rwrist']],  # right wrist
    [7, MPII_BODY_PARTS['lhip']],  # left hip
    [8, MPII_BODY_PARTS['rhip']],  # right hip
    [9, MPII_BODY_PARTS['lknee']],  # left knee
    [10, MPII_BODY_PARTS['rknee']],  # right knee
    [11, MPII_BODY_PARTS['lankle']],  # left ankle
    [12, MPII_BODY_PARTS['rankle']]  # right ankle
])


def get_movenet_keypoints(mpii_annorect, h_rescaling_factor=1., w_rescaling_factor=1., add_visibility=True):

    keypoints = []

    # use mpii's head rectangle center as head keypoint
    head_x = (mpii_annorect['x1'] + mpii_annorect['x2']) / 2
    head_x *= w_rescaling_factor

    head_y = (mpii_annorect['y1'] + mpii_annorect['y2']) / 2
    head_y *= h_rescaling_factor

    if add_visibility:
        keypoints.append([head_x, head_y, 2])
    else:
        keypoints.append([head_x, head_y])

    for ind in MOVENET_13_TO_MPII_INDICES[1:]:
        mpii_keypoint_ind = ind[1]
        visibility = 0
        for point in mpii_annorect['annopoints']['point']:

            if point['id'] != mpii_keypoint_ind:
                continue

            if point['is_visible'] == 1:
                visibility = 2
            else:
                visibility = 1

            if add_visibility:
                keypoints.append([point['x'] * w_rescaling_factor,
                                  point['y'] * h_rescaling_factor,
                                  visibility])
            else:
                keypoints.append([point['x'] * w_rescaling_factor,
                                  point['y'] * h_rescaling_factor])

            break

        if visibility == 0:  # keypoint has not been annotated
            if add_visibility:
                keypoints.append([0, 0, visibility])
            else:
                keypoints.append([0, 0])

    return keypoints


def get_mpii_other_centers(ind_to_exclude, mpii_annorect, h_rescaling_factor=1., w_rescaling_factor=1.):

    centers = []
    for ai, ann in enumerate(mpii_annorect):
        if ai == ind_to_exclude:
            continue
        centers.append([ann['objpos']['x'] * w_rescaling_factor,
                        ann['objpos']['y'] * h_rescaling_factor])
    return centers


def get_mpii_other_keypoints(ind_to_exclude, mpii_annorect, h_rescaling_factor=1., w_rescaling_factor=1.):

    keypoints = []
    for ai, ann in enumerate(mpii_annorect):
        if ai == ind_to_exclude:
            continue

        # add keypoint without visibility element (not needed in 'other_keypoints')
        keypoints.append(get_movenet_keypoints(ann, h_rescaling_factor, w_rescaling_factor, add_visibility=False))
    return keypoints


def mpii_to_movenet(poses_mpii, images_folder, image_name):

    poses_movenet = []
    for ai, p_mpii in enumerate(poses_mpii):

        try:
            p_movenet = dict()
            p_movenet['img_name'] = image_name

            img_path = images_folder / image_name
            img = cv2.imread(str(img_path.resolve()))
            h, w, _ = img.shape
            h_rescaling_factor = 1 / h
            w_rescaling_factor = 1 / w

            p_movenet['keypoints'] = get_movenet_keypoints(p_mpii, h_rescaling_factor, w_rescaling_factor)
            p_movenet['center'] = [p_mpii['objpos']['x'] * w_rescaling_factor,
                                   p_mpii['objpos']['y'] * h_rescaling_factor]
            # p_movenet['bbox'] = get_movenet_bbox(p_movenet['keypoints'])
            p_movenet['other_centers'] = get_mpii_other_centers(ai, poses_mpii, h_rescaling_factor, w_rescaling_factor)
            p_movenet['other_keypoints'] = get_mpii_other_keypoints(ai, poses_mpii, h_rescaling_factor, w_rescaling_factor)
        except:
            continue

        poses_movenet.append(p_movenet)

    return poses_movenet


def export_to_tos(data_ann, image_folder, output_folder, gaussian_blur_k_size, gaussian_blur_sigma, canny_low_th,
                  canny_high_th, canny_aperture, canny_l2_grad, salt_pepper_low_th, salt_pepper_high_th):

    iterator = mpii_parse.MPIIIterator(data_ann, image_folder)

    tos = EROSSynthetic(gaussian_blur_k_size, gaussian_blur_sigma, canny_low_th, canny_high_th,
                        canny_aperture, canny_l2_grad, salt_pepper_low_th, salt_pepper_high_th)

    poses = []
    for fi, (img, poses_ann, img_name) in enumerate(tqdm(iterator)):

        # skip images with more than three people (as in movenet's code)
        if len(poses_ann) > 3:
            continue

        if img is None:
            print(f'image {img_name} does not exist')
            continue

        movenet_poses = mpii_to_movenet(poses_ann, image_folder, img_name)

        if len(movenet_poses) == 0:
            print(f'no annotations for image {img_name}')
            continue

        poses.extend(movenet_poses)

        frame = tos.get_frame(img)
        file_path = output_folder / img_name
        if not cv2.imwrite(str(file_path), frame):
            print(f'could not save image to {str(file_path)}')

    output_json = output_folder / 'poses.json'
    with open(str(output_json.resolve()), 'w') as f:
        json.dump(poses, f, ensure_ascii=False)



def main(args):

    data_ann_path = pathlib.Path(args.a)
    data_ann_path = pathlib.Path(data_ann_path.resolve())
    data_ann = mat_utils.loadmat(str(data_ann_path))

    img_folder_path = pathlib.Path(args.i)
    img_folder_path = pathlib.Path(img_folder_path.resolve())

    output_folder_path = pathlib.Path(args.o)
    output_folder_path = pathlib.Path(output_folder_path.resolve())
    output_folder_path.mkdir(parents=True, exist_ok=True)

    export_to_tos(data_ann, img_folder_path, output_folder_path,
                  gaussian_blur_k_size=args.gk, gaussian_blur_sigma=args.gs,
                  canny_low_th=args.cl, canny_high_th=args.ch, canny_aperture=args.ca, canny_l2_grad=args.cg,
                  salt_pepper_low_th=args.sl, salt_pepper_high_th=args.sh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', help='path to annotations file')
    parser.add_argument('-i', help='path to images folder')
    parser.add_argument('-o', help='path to output folder')

    # synthetic TOS params

    # canny edge detector params
    parser.add_argument('-gk', help='gaussian blur kernel size', type=int, default=5)
    parser.add_argument('-gs', help='gaussian blur sigma', type=int, default=0)
    parser.add_argument('-cl', help='canny edge low threshold', type=int, default=0)
    parser.add_argument('-ch', help='canny edge high threshold', type=int, default=1000)
    parser.add_argument('-ca', help='canny edge aperture', type=int, default=5)
    parser.add_argument('-cg', help='canny edge l2 gradient flag', dest='cg', action='store_true')
    parser.set_defaults(cg=False)

    # salt and pepper params
    parser.add_argument('-sl', help='salt and pepper low threshold', type=int, default=5)
    parser.add_argument('-sh', help='salt and pepper high threshold', type=int, default=250)

    args = parser.parse_args()

    main(args)
