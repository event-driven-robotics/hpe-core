"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author:
    Nicolo' Carissimi

LICENSE GOES HERE
"""

import argparse
import cv2
import pathlib

import datasets.dhp19.utils.constants as dhp19_const
import datasets.dhp19.utils.mat_files as mat_utils
import datasets.dhp19.utils.parsing as dhp19_parse

from datasets.tos_utils import EROS


def export_to_eros(data_dvs, data_vicon, cam_id, events_window_size, output_folder, proj_mat_folder=None,
                   events_window_stride=None, eros_k_size=7, gaussian_blur_k_size=5, gaussian_blur_sigma=0):

    iterator = dhp19_parse.Dhp19Iterator(data_dvs=data_dvs,
                                         data_vicon=data_vicon,
                                         cam_id=cam_id,
                                         proj_mat_folder=proj_mat_folder,
                                         window_size=events_window_size,
                                         stride=events_window_stride)

    if eros_k_size % 2 != 0:
        eros_k_size += 1

    eros = EROS(kernel_size=eros_k_size,
                frame_width=dhp19_const.DHP19_SENSOR_WIDTH,
                frame_height=dhp19_const.DHP19_SENSOR_HEIGHT)

    for fi, (events, poses_2d) in enumerate(iterator):
        for ei in range(len(events)):
            eros.update(vx=int(events[ei, 1]), vy=int(events[ei, 2]))
        frame = eros.get_frame()

        # apply gaussian filter
        kernel = (gaussian_blur_k_size, gaussian_blur_k_size)
        frame = cv2.GaussianBlur(frame, kernel, gaussian_blur_sigma)

        file_path = output_folder / f'frame_{fi}.png'
        cv2.imwrite(str(file_path), frame)


def main(args):
    # read dvs file
    data_dvs_path = pathlib.Path(args.e)
    data_dvs_path = pathlib.Path(data_dvs_path.resolve())
    data_dvs = mat_utils.loadmat(str(data_dvs_path))

    # read vicon file
    data_vicon_path = pathlib.Path(args.v)
    data_vicon_path = pathlib.Path(data_vicon_path.resolve())
    data_vicon = mat_utils.loadmat(str(data_vicon_path))

    cam_id = args.c

    # projection matrix folder
    proj_matrices_folder = None
    if args.p:
        proj_matrices_folder = pathlib.Path(args.p)
        proj_matrices_folder = pathlib.Path(proj_matrices_folder.resolve())

    # output folder
    export_folder = pathlib.Path(args.o) / str(cam_id)
    export_folder = pathlib.Path(export_folder.resolve())
    export_folder.mkdir(parents=True, exist_ok=True)

    export_to_eros(data_dvs, data_vicon, cam_id, args.window_size, export_folder, proj_matrices_folder,
                   args.window_stride)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help='path to .mat DVS file')
    parser.add_argument('-v', help='path to .mat Vicon file')
    parser.add_argument('-w', '--window_size',
                        help=f'approximate number of events used to compute an event frame (default value for DHP19 is {dhp19_const.DHP19_CAM_FRAME_EVENTS_NUM})',
                        default=dhp19_const.DHP19_CAM_FRAME_EVENTS_NUM)
    parser.add_argument('-s', '--window_stride', type=int, required=False)
    parser.add_argument('-c', help='camera id', type=int)
    parser.add_argument('-o', help='path to output folder')
    parser.add_argument('-p', help='path to projection matrices folder', required=False)
    parser.add_argument('-td', '--two_dimensional', dest='two_dimensional',
                        help='flag specifying if 2D poses will be extracted; if not specified, 3D poses will be extracted',
                        action='store_true')
    parser.set_defaults(two_dimensional=False)
    args = parser.parse_args()

    main(args)
