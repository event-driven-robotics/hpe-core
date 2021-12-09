#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Event-driven Perception for Robotics
Authors:
    Franco Di Pietro
    Nicolo' Carissimi

LICENSE GOES HERE
"""

import argparse
import numpy as np
import pathlib

import datasets.dhp19.utils.constants as dhp19_const
import datasets.dhp19.utils.mat_files as mat_utils
import datasets.dhp19.utils.parsing as dhp19_parse

from bimvee import exportIitYarp


def export_to_yarp(data_dvs, data_vicon, projection_mat_folder, window_size, output_folder):

    # build events container
    container = {'info': {}, 'data': {}}
    start_time = data_dvs['out']['extra']['startTime']
    for cn in range(dhp19_const.DHP19_CAM_NUM):
        data = data_dvs['out']['data'][f'cam{cn}']
        data['dvs']['x'] = data['dvs']['x'] - 1 - dhp19_const.DHP19_SENSOR_WIDTH * cn
        data['dvs']['y'] = data['dvs']['y'] - 1
        data['dvs']['ts'] = (data['dvs']['ts'] - start_time) * 1e-6
        data['dvs']['pol'] = np.array(data['dvs']['pol'], dtype=bool)
        container['data'][f'ch{cn}'] = data

    avg_poses_3d, avg_poses_ts = dhp19_parse.extract_3d_poses(data_dvs, data_vicon, window_size)

    for cn in range(dhp19_const.DHP19_CAM_NUM):
        if projection_mat_folder:

            projection_mat = dhp19_parse.get_projection_matrix(cn, projection_mat_folder)
            avg_poses_2d, joints_mask = dhp19_parse.project_poses_to_2d(avg_poses_3d, np.transpose(projection_mat))

            # convert avg_poses_2d to dictionary
            container['data'][f'ch{cn}']['skeleton'] = dict()
            for joint_type in dhp19_parse.DHP19_BODY_PARTS.keys():
                container['data'][f'ch{cn}']['skeleton'][joint_type] = avg_poses_2d[:, dhp19_parse.DHP19_BODY_PARTS[joint_type], :]
        else:
            # convert avg_poses_3d to dictionary
            for joint_type in dhp19_parse.DHP19_BODY_PARTS.keys():
                container['data'][f'ch{cn}']['skeleton'][joint_type] = avg_poses_3d[:, dhp19_parse.DHP19_BODY_PARTS[joint_type], :]
        container['data'][f'ch{cn}']['skeleton']['ts'] = avg_poses_ts

    exportIitYarp.exportIitYarp(container, exportFilePath=str(output_folder.resolve()), protectedWrite=True)


def main(args):

    # read dvs file
    data_dvs_path = pathlib.Path(args.e)
    data_dvs_path = pathlib.Path(data_dvs_path.resolve())
    data_dvs = mat_utils.loadmat(str(data_dvs_path))

    # read vicon file
    data_vicon_path = pathlib.Path(args.v)
    data_vicon_path = pathlib.Path(data_vicon_path.resolve())
    data_vicon = mat_utils.loadmat(str(data_vicon_path))

    # projection matrix folder
    proj_matrices_folder = pathlib.Path(args.p)
    proj_matrices_folder = pathlib.Path(proj_matrices_folder.resolve())

    # output folder
    export_folder = pathlib.Path(args.o)
    export_folder = pathlib.Path(export_folder.resolve())

    export_to_yarp(data_dvs, data_vicon, proj_matrices_folder, args.window_size, export_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help='path to .mat DVS file')
    parser.add_argument('-v', help='path to .mat Vicon file')
    parser.add_argument('-p', help='path to projection matrices folder')
    parser.add_argument('-w', '--window_size',
                        help=f'approximate number of events used to compute an event frame (default value for DHP19 is {dhp19_const.DHP19_CAM_FRAME_EVENTS_NUM})',
                        default=dhp19_const.DHP19_CAM_FRAME_EVENTS_NUM)
    parser.add_argument('-o', help='path to output folder')
    args = parser.parse_args()

    main(args)
