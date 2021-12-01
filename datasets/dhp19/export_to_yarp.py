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
import os
import pathlib
import sys

import datasets.dhp19.utils.constants as dhp19_const
import datasets.dhp19.utils.mat_files as mat_utils

# add bimvee and mustard paths
bimvee_path = os.environ.get('BIMVEE_PATH')
sys.path.insert(0, bimvee_path)
mustard_path = os.environ.get('MUSTARD_PATH')
sys.path.insert(0, mustard_path)

from bimvee import exportIitYarp


def export_to_yarp(input_file_path, output_folder):

    dataDvs = mat_utils.loadmat(str(input_file_path))

    # build container
    container = {'info': {}, 'data': {}}
    start_time = dataDvs['out']['extra']['startTime']
    for cn in range(dhp19_const.DHP19_CAM_NUM):
        container['data'][f'ch{cn}'] = dataDvs['out']['data'][f'cam{cn}']
        container['data'][f'ch{cn}']['dvs']['x'] = container['data'][f'ch{cn}']['dvs']['x'] - 1 - dhp19_const.DHP19_SENSOR_WIDTH * cn
        container['data'][f'ch{cn}']['dvs']['y'] = container['data'][f'ch{cn}']['dvs']['y'] - 1
        container['data'][f'ch{cn}']['dvs']['ts'] = (container['data'][f'ch{cn}']['dvs']['ts'] - start_time) * 1e-6
        container['data'][f'ch{cn}']['dvs']['pol'] = np.array(container['data'][f'ch{cn}']['dvs']['pol'], dtype=bool)

    exportIitYarp.exportIitYarp(container, exportFilePath=str(output_folder / input_file_path.stem), protectedWrite=True)


def main(args):

    raw_data_folder = pathlib.Path(args.i)
    raw_data_folder = pathlib.Path(raw_data_folder.resolve())

    export_folder = pathlib.Path(args.o)
    export_folder = pathlib.Path(export_folder.resolve())

    for raw_file in raw_data_folder.glob('*.mat'):
        export_to_yarp(raw_file, export_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to raw .mat DVS files', default='/home/fdipietro/hpe-data/DVS')
    parser.add_argument('-o', help='path to output folder', default='/home/fdipietro/hpe-data/yarp')
    args = parser.parse_args()

    main(args)
