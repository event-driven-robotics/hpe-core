
import argparse

import cv2
import math
import numpy as np
import pathlib

from bimvee.importIitYarp import importIitYarp
from tqdm import tqdm

import datasets.utils.parsing as parsing

from datasets.utils.constants import HPECoreSkeleton
from datasets.utils.events_representation import EROS


def get_keypoints(pose, h_frame=1, w_frame=1, add_visibility=True):
    keypoints = []
    for k in pose:
        if add_visibility:
            k_scaled = [k[0] / w_frame, k[1] / h_frame, 2]
        else:
            k_scaled = [k[0] / w_frame, k[1] / h_frame]
        keypoints.extend(k_scaled)
    return keypoints


def get_torso_length(pose):
    shoulder_mean = np.mean([pose[HPECoreSkeleton.KEYPOINTS_MAP['left_shoulder'], :],
                             pose[HPECoreSkeleton.KEYPOINTS_MAP['right_shoulder'], :]], axis=0)
    hip_mean = np.mean([pose[HPECoreSkeleton.KEYPOINTS_MAP['left_hip'], :],
                        pose[HPECoreSkeleton.KEYPOINTS_MAP['right_hip'], :]], axis=0)
    # torso_dist = math.dist(shoulder_mean, hip_mean)
    torso_dist = np.linalg.norm([shoulder_mean, hip_mean])

    return torso_dist


def yarp_to_eros(data_dvs, data_skeleton, output_folder_path, frame_width, frame_height,
                 eros_k_size=7, gaussian_blur_k_size=5, gaussian_blur_sigma=0, file_name_prefix=None):

    iterator = parsing.YarpHPEIterator(data_dvs['data']['left']['dvs'], data_skeleton)
    eros = EROS(kernel_size=eros_k_size, frame_width=frame_width, frame_height=frame_height)

    # poses_movenet = []
    for fi, (events, skeleton, skeleton_ts) in tqdm(enumerate(iterator)):
        for ei in range(len(events)):
            eros.update(vx=int(events[ei, 1]), vy=int(events[ei, 2]))
        frame = eros.get_frame()

        # apply gaussian filter
        kernel = (gaussian_blur_k_size, gaussian_blur_k_size)
        frame = cv2.GaussianBlur(frame, kernel, gaussian_blur_sigma)

        if fi == 0:  # Almost empty image, not beneficial for training
            # kps_old = get_keypoints(pose, frame_height, frame_width)
            continue

        if file_name_prefix:
            file_path = output_folder_path / f'{file_name_prefix}_frame_{fi:06}.png'
        else:
            file_path = output_folder_path / f'frame_{fi:06}.png'
        cv2.imwrite(str(file_path.resolve()), frame)


def create_annotation(skeleton, image_name, image_width, image_height):

    sample_anno = dict()
    sample_anno['image_name'] = image_name
    sample_anno['image_height'] = image_height
    sample_anno['image_width'] = image_width
    sample_anno['keypoints'] = skeleton.to_list()
    sample_anno['torso_size'] = get_torso_length(skeleton)
    sample_anno['head_size'] = get_head_size(skeleton)

    return sample_anno


def parse_skeleton(data_dvs, data_skeleton, output_folder_path, frame_width, frame_height, file_name_prefix=None):

    iterator = parsing.YarpHPEIterator(data_dvs['data']['left']['dvs'], data_skeleton)

    poses_movenet = []
    for fi, (_, skeleton, ts) in enumerate(iterator):

        if fi == 0:  # Almost empty image, not beneficial for training
            kps_old = get_keypoints(skeleton, frame_height, frame_width)
            continue

        sample_anno = dict()
        sample_anno['img_name'] = file_name
        sample_anno['keypoints'] = get_keypoints(skeleton, frame_height, frame_width)
        sample_anno['center'] = get_center(skeleton, frame_height, frame_width)
        sample_anno['torso_size'] = get_torso_length(skeleton, frame_height, frame_width)
        sample_anno['keypoints_prev'] = kps_old
        sample_anno['other_centers'] = []
        sample_anno['other_keypoints'] = [[] for _ in range(17)]
        sample_anno['head_size'] = []
        sample_anno['head_size_scaled'] = []
        sample_anno['original_sample'] = file_name_prefix

        poses_movenet.append(sample_anno)

        kps_old = sample_anno['keypoints']

    return poses_movenet


def main(args):

    # import skeletons
    skeleton_folder_path = pathlib.Path(args.s)
    skeleton_folder_path = skeleton_folder_path.resolve()
    skeleton_file_path = skeleton_folder_path / 'data.log'
    data_skeleton = parsing.import_yarp_skeleton_data(skeleton_file_path)

    # import events
    dvs_folder_path = pathlib.Path(args.e)
    dvs_folder_path = dvs_folder_path.resolve()
    data_dvs = importIitYarp(filePathOrName=str(dvs_folder_path))

    # create output folder
    output_folder_path = pathlib.Path(args.o)
    output_folder_path = output_folder_path.resolve()
    output_folder_path.mkdir(parents=True, exist_ok=True)

    yarp_to_eros(data_dvs, data_skeleton, output_folder_path, args.fw, args.fh, file_name_prefix='')
    # yarp_to_hpecore_skeleton(data_dvs, data_skeleton, output_folder_path, args.w, args.h, file_name_prefix='')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help='path to events yarp folder', required=True)
    parser.add_argument('-s', help='path to skeletons yarp folder', required=True)
    parser.add_argument('-fw', help='frame width', type=int, required=True)
    parser.add_argument('-fh', help='frame height', type=int, required=True)
    parser.add_argument('-o', help='path to output folder', required=True)
    args = parser.parse_args()
    main(args)
