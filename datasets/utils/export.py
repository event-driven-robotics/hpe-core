
import numpy

from pathlib import Path
from typing import Optional


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


def export_skeletons_to_yarp(skeletons: numpy.array, timestamps: numpy.array, output_dir: Path,
                             cam: int, head_sizes: Optional[numpy.array] = None, torso_sizes: Optional[numpy.array] = None):

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
        f.write(f'[0.0] /file/ch{cam}GTskeleton:o [connected]\n')
