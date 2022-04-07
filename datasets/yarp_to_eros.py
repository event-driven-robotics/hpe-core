
import argparse

import cv2
import json
import pathlib

from bimvee.importIitYarp import importIitYarp
from tqdm import tqdm

import datasets.utils.export as export
import datasets.utils.parsing as parsing

from datasets.utils.events_representation import EROS


def yarp_to_eros(data_dvs, data_skeleton, output_folder_path, frame_width, frame_height,
                 eros_k_size=7, gaussian_blur_k_size=5, gaussian_blur_sigma=0, frames_name_prefix=None):

    iterator = parsing.YarpHPEIterator(data_dvs['data']['left']['dvs'], data_skeleton)
    eros = EROS(kernel_size=eros_k_size, frame_width=frame_width, frame_height=frame_height)

    hpecore_skeletons = []
    for fi, (events, skeleton, skeleton_ts, head_size, torso_size) in tqdm(enumerate(iterator)):
        for ei in range(len(events)):
            eros.update(vx=int(events[ei, 1]), vy=int(events[ei, 2]))
        frame = eros.get_frame()

        # apply gaussian filter
        kernel = (gaussian_blur_k_size, gaussian_blur_k_size)
        frame = cv2.GaussianBlur(frame, kernel, gaussian_blur_sigma)

        if fi == 0:  # Almost empty image, not beneficial for training
            # kps_old = get_keypoints(pose, frame_height, frame_width)
            continue

        # save eros frame
        if frames_name_prefix:
            frame_name = f'{frames_name_prefix}_frame_{fi:06}.png'
        else:
            frame_name = f'frame_{fi:06}.png'
        frame_path = output_folder_path / frame_name
        cv2.imwrite(str(frame_path.resolve()), frame)

        hpecore_skeletons.append(export.skeleton_to_dict(skeleton, frame_name, frame_width, frame_height, head_size, torso_size))

    # save skeletons to a json files
    skeletons_json_path = output_folder_path / 'skeletons.json'
    with open(str(skeletons_json_path.resolve()), 'w') as f:
        json.dump(hpecore_skeletons, f, ensure_ascii=False)


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

    yarp_to_eros(data_dvs, data_skeleton, output_folder_path, args.fw, args.fh, frames_name_prefix='')
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
