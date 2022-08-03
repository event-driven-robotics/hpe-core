
import numpy
from pathlib import Path
from typing import Optional, Dict
import os
import argparse
import cv2
import math
from datasets.utils.constants import HPECoreSkeleton
from datasets.h36m.utils.parsing import hpecore_to_movenet


def skeleton_to_dict(skeleton, image_name, image_width, image_height, head_size, torso_size):

    sample_anno = dict()
    sample_anno['image_name'] = image_name
    sample_anno['image_height'] = image_height
    sample_anno['image_width'] = image_width
    sample_anno['keypoints'] = skeleton.to_list()
    sample_anno['torso_size'] = torso_size
    sample_anno['head_size'] = head_size

    return sample_anno


def skeleton_to_yarp_row(counter: int, timestamp: float, skeleton: numpy.array, head_size: float = -1.0, torso_size: float = -1.0) -> str:

    skeleton_str = numpy.array2string(skeleton.reshape(-1), max_line_width=1000, precision=0, separator=' ')
    skeleton_str = skeleton_str[1:-1]

    row = f'{counter} {timestamp:06} SKLT ({skeleton_str}) {head_size} {torso_size}'

    return row


def export_list_to_yarp(elems: list, info: str, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    # write skeleton coordinates, head size and torso for every line
    data_file = output_dir / 'data.log'
    with open(str(data_file.resolve()), 'w') as f:
        for elem in range(len(elems)):
            f.write(f'{str(elem)}\n')

    info_file = output_dir / 'info.log'
    with open(str(info_file.resolve()), 'w') as f:
        f.write(info)


def export_skeletons_to_yarp(skeletons: numpy.array, timestamps: numpy.array, output_dir: Path,
                             channel: int, head_sizes: Optional[numpy.array] = None, torso_sizes: Optional[numpy.array] = None):

    assert len(skeletons.shape) == 3 and (skeletons.shape[2] == 2 or skeletons.shape[2] == 3), \
        'skeleton shape must be (n, joints_num, 2) or (n, joints_num, 3)'

    output_dir.mkdir(parents=True, exist_ok=True)

    # write skeleton coordinates, head size and torso for every line
    data_file = output_dir / 'data.log'
    with open(str(data_file.resolve()), 'w') as f:
        for si in range(len(skeletons)):

            hs = -1.0 if head_sizes is None else head_sizes[si]
            ts = -1.0 if torso_sizes is None else torso_sizes[si]
            f.write(f'{skeleton_to_yarp_row(si, timestamps[si], skeletons[si, :], hs, ts)}\n')

    info_file = output_dir / 'info.log'
    with open(str(info_file.resolve()), 'w') as f:
        f.write('Type: Bottle;\n')
        f.write(f'[0.0] /file/ch{channel}GTskeleton:o [connected]\n')


def format_crop_file(crop_lines):
    # Convert the file containing cropping data into a dictionary
    crop_dict = {}
    for line in crop_lines:
        file, crop_str = line.split(' ')
        l, r, t, b = crop_str.split(',')
        crop_dict[file] = {'left': int(l), 'right': int(r), 'top': int(t), 'bottom': int(b)}
    return crop_dict


def crop_frame(frame, crop):
    # crop a frame based on values from dictonary element crop.
    w,h,_ = frame.shape
    if crop is not None:
        frame = frame[crop['top'] + 1:w-crop['bottom'], crop['left'] + 1:h-crop['right'], :]
    return frame


def crop_pose(sklt, crop):
    # adjust GT skeleton value due to cropping based on dictonary element crop.
    if crop is not None:
        for i in range(len(sklt)):
            sklt[i, 0] -= crop['left']
            sklt[i, 1] -= crop['top']
    return sklt

def ensure_location(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def checkframecount(video_file_name, gt_file_name):

    vid = cv2.VideoCapture(video_file_name)
    vid_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.release()

    num_lines = sum(1 for line in open(gt_file_name))

    if vid_frames == 0:
        print("no video frames")
        return False
    if num_lines == 0:
        print("no skeleton files")
        return False
    if vid_frames != num_lines:
        print("not correctly processed", vid_frames, num_lines)
        return False
    return True


def get_movenet_keypoints(pose: Dict, h_frame=1, w_frame=1, add_visibility=True):

    keypoints = []
    pose_hpecore = numpy.zeros([len(HPECoreSkeleton.KEYPOINTS_MAP),2], float)
    for key, value in HPECoreSkeleton.KEYPOINTS_MAP.items():
        pose_hpecore[value,:] = pose[key][:]
    pose_movenet = hpecore_to_movenet(pose_hpecore)
    for k in pose_movenet:
        if add_visibility:
            k_scaled = [k[0] / w_frame, k[1] / h_frame, 2]
        else:
            k_scaled = [k[0] / w_frame, k[1] / h_frame]
        keypoints.extend(k_scaled)
    return keypoints


def get_torso_length(pose, h_frame=1, w_frame=1):
    k = {}
    # k['left_shoulder']=pose[1,:]
    # k['right_shoulder']=pose[2,:]
    # k['left_hip']=pose[7,:]
    # k['right_hip']=pose[8,:]

    k['shoulder_mean'] = numpy.mean(pose[1:3, :], axis=0)
    k['hip_mean'] = numpy.mean(pose[7:9, :], axis=0)
    k['shoulder_mean'] = k['shoulder_mean'][0] / w_frame, k['shoulder_mean'][1] / h_frame
    k['hip_mean'] = k['hip_mean'][0] / w_frame, k['hip_mean'][1] / h_frame
    k['torso_dist'] = math.dist(k['shoulder_mean'], k['hip_mean'])

    return k['torso_dist']


def get_center(pose, h_frame=1, w_frame=1):
    # x_cen = np.mean([min(pose[:, 0]), max(pose[:, 0])]) / w_frame
    # y_cen = np.mean([min(pose[:, 1]), max(pose[:, 1])]) / h_frame
    pose_hpecore = numpy.zeros([len(HPECoreSkeleton.KEYPOINTS_MAP),2], float)
    for key, value in HPECoreSkeleton.KEYPOINTS_MAP.items():
        pose_hpecore[value,:] = pose[key][:]
    x_cen = numpy.mean(pose_hpecore[:, 0]) / w_frame
    y_cen = numpy.mean(pose_hpecore[:, 1]) / h_frame
    return [x_cen, y_cen]