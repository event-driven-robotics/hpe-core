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

from bimvee import exportIitYarp

import datasets.dhp19.utils.constants as dhp19_const
import datasets.dhp19.utils.parsing as dhp19_parse
import datasets.utils.export as hpecore_export
import datasets.utils.mat_files as mat_utils

from datasets.utils.constants import HPECoreSkeleton


def export_to_yarp(data_dvs: dict, data_vicon: dict, projection_mat_folder: pathlib.Path, output_folder: pathlib.Path):

    ###############
    # export events
    ###############

    # create yarp container
    container = {'info': {}, 'data': {}}

    start_time = data_dvs['out']['extra']['startTime']
    for cn in range(dhp19_const.DHP19_CAM_NUM):

        data = data_dvs['out']['data'][f'cam{cn}']

        # parse event coordinates
        data['dvs']['x'] = data['dvs']['x'] - 1 - dhp19_const.DHP19_SENSOR_WIDTH * cn
        data['dvs']['y'] = data['dvs']['y'] - 1

        # normalize timestamp
        data['dvs']['ts'] = (data['dvs']['ts'] - start_time) * 1e-6

        # convert polarity to an array of booleans
        data['dvs']['pol'] = np.array(data['dvs']['pol'], dtype=bool)

        container['data'][f'ch{cn}'] = data

    exportIitYarp.exportIitYarp(container, exportFilePath=str(output_folder.resolve()), protectedWrite=True)

    ##################
    # export skeletons
    ##################

    # create array of timestamps for the poses
    dt = 10000
    poses_ts = np.arange(data_dvs['out']['extra']['ts'][0] - start_time, data_dvs['out']['extra']['ts'][-1] - start_time + dt,
                    dt) * 1e-6  # Vicon timestams @ 100Hz
    diff = len(poses_ts) - data_vicon['XYZPOS']['head'].shape[0]
    if diff > 0:
        poses_ts = poses_ts[:-diff]

    poses_3d = extract_3d_poses(data_vicon)
    poses_3d = dhp19_parse.dhp19_to_hpecore_skeletons(poses_3d)

    for cn in range(dhp19_const.DHP19_CAM_NUM):
        if projection_mat_folder:
            projection_mat = dhp19_parse.get_projection_matrix(cn, projection_mat_folder)
            poses_2d, joints_mask = dhp19_parse.project_poses_to_2d(poses_3d, np.transpose(projection_mat))

            torsos_size = HPECoreSkeleton.compute_torso_sizes(poses_2d)

            # export skeletons for current camera
            output_folder_cam = output_folder / f'ch{cn}skeleton'
            hpecore_export.export_skeletons_to_yarp(poses_2d, poses_ts, output_folder_cam, cn, torso_sizes=torsos_size)
        else:
            torsos_size = HPECoreSkeleton.compute_torso_sizes(poses_3d)

            # export skeletons for current camera
            output_folder_cam = output_folder / f'ch{cn}skeleton'
            hpecore_export.export_skeletons_to_yarp(poses_3d, poses_ts, output_folder_cam, cn, torso_sizes=torsos_size)


def extract_3d_poses(data_vicon):
    """
    ...

    Parameters:
        data_vicon (dict): dictionary containing Vicon data, provided by DHP19 dataset
    Returns:
        a numpy array with shape (num_of_poses, num_of_dhp19_joints, 3) containing the 3d poses
    """

    poses_3d = np.zeros(shape=(len(data_vicon['XYZPOS']['head']), len(dhp19_parse.DHP19_BODY_PARTS), 3))

    for body_part in dhp19_parse.DHP19_BODY_PARTS:
        coords = data_vicon['XYZPOS'][body_part]
        poses_3d[:, dhp19_parse.DHP19_BODY_PARTS[body_part], :] = coords

    return poses_3d


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

    export_to_yarp(data_dvs, data_vicon, proj_matrices_folder, export_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help='path to .mat DVS file')
    parser.add_argument('-v', help='path to .mat Vicon file')
    parser.add_argument('-p', help='path to projection matrices folder')
    parser.add_argument('-o', help='path to output folder')
    args = parser.parse_args()

    main(args)
