
import numpy

from pathlib import Path
from typing import Optional
import os


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
