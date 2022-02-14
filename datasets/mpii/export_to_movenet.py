
import argparse
import cv2
import json
import numpy as np
import pathlib

from tqdm import tqdm

import datasets.utils.mat_files as mat_utils
import datasets.mpii.utils.parsing as mpii_parse

from datasets.mpii.utils.parsing import MPII_BODY_PARTS


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


def _compute_bbox(mpii_annorect, h_rescaling_factor=1., w_rescaling_factor=1.):

    # parse all keypoints
    keypoints = np.zeros((len(mpii_annorect['annopoints']['point']), 2))
    for pi, point in enumerate(mpii_annorect['annopoints']['point']):
        keypoints[pi, 0] = point['x'] * w_rescaling_factor
        keypoints[pi, 1] = point['y'] * h_rescaling_factor

    # compute bbox
    bbox_min_x = np.min(keypoints[:, 0])
    bbox_max_x = np.max(keypoints[:, 0])
    bbox_min_y = np.min(keypoints[:, 1])
    bbox_max_y = np.max(keypoints[:, 1])

    # expand to a larger square
    cx = (bbox_min_x + bbox_max_x) / 2
    cy = (bbox_min_y + bbox_max_y) / 2
    half_size = ((bbox_max_x - bbox_min_x) + (bbox_max_y - bbox_min_y)) / 2
    bbox_min_x = int(cx - half_size)
    bbox_max_x = int(cx + half_size)
    bbox_min_y = int(cy - half_size)
    bbox_max_y = int(cy + half_size)

    return [bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y]


def _compute_head_size(mpii_annorect, h_rescaling_factor=1., w_rescaling_factor=1.):
    sc_bias = 0.6  # 0.8 * 0.75 constant used in mpii matlab code for computing the head size
    head_size = sc_bias * np.linalg.norm([(mpii_annorect['x2'] - mpii_annorect['x1']) * w_rescaling_factor,
                                          (mpii_annorect['y2'] - mpii_annorect['y1']) * h_rescaling_factor])
    return head_size


def _get_movenet_keypoints(mpii_annorect, h_rescaling_factor=1., w_rescaling_factor=1., x_shift=0, y_shift=0, add_visibility=True):

    keypoints = []

    # use mpii's head rectangle center as head keypoint
    head_x = ((mpii_annorect['x1'] + mpii_annorect['x2']) / 2) + x_shift
    head_x *= w_rescaling_factor

    head_y = ((mpii_annorect['y1'] + mpii_annorect['y2']) / 2) + y_shift
    head_y *= h_rescaling_factor

    if add_visibility:
        keypoints.extend([head_x, head_y, 2])
    else:
        keypoints.extend([head_x, head_y])

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
                keypoints.extend([(point['x'] + x_shift) * w_rescaling_factor,
                                  (point['y'] + y_shift) * h_rescaling_factor,
                                  visibility])
            else:
                keypoints.extend([(point['x'] + x_shift) * w_rescaling_factor,
                                  (point['y'] + y_shift) * h_rescaling_factor])

            break

        if visibility == 0:  # keypoint has not been annotated
            if add_visibility:
                keypoints.extend([0, 0, visibility])
            else:
                keypoints.extend([0, 0])

    return keypoints


def _get_movenet_other_centers(ind_to_exclude, mpii_annorect, h_rescaling_factor=1., w_rescaling_factor=1., x_shift=0, y_shift=0):

    centers = []
    for ai, ann in enumerate(mpii_annorect):
        if ai == ind_to_exclude:
            continue

        x = (ann['objpos']['x'] + x_shift) * w_rescaling_factor
        y = (ann['objpos']['y'] + y_shift) * h_rescaling_factor
        if _is_coord_inside_boundaries(x) and _is_coord_inside_boundaries(y):
            centers.append([x, y])

    return centers


def _get_movenet_other_keypoints(ind_to_exclude, mpii_annorect, h_rescaling_factor=1., w_rescaling_factor=1., x_shift=0, y_shift=0):

    keypoints = [[] for _ in range(len(MOVENET_13_TO_MPII_INDICES))]

    for ai, ann in enumerate(mpii_annorect):
        if ai == ind_to_exclude:
            continue

        # calculate head keypoint
        head_x = ((ann['x1'] + ann['x2']) / 2) + x_shift
        head_y = ((ann['y1'] + ann['y2']) / 2) + y_shift

        # append head keypoint to list
        x = head_x * w_rescaling_factor
        y = head_y * h_rescaling_factor
        if _is_coord_inside_boundaries(x) and _is_coord_inside_boundaries(y):
            keypoints[MOVENET_13_TO_MPII_INDICES[0][0]].append([x, y])

        for ind in MOVENET_13_TO_MPII_INDICES[1:]:
            mpii_keypoint_ind = ind[1]
            for point in ann['annopoints']['point']:

                if point['id'] != mpii_keypoint_ind:
                    continue

                x = (point['x'] + x_shift) * w_rescaling_factor
                y = (point['y'] + y_shift) * h_rescaling_factor
                if _is_coord_inside_boundaries(x) and _is_coord_inside_boundaries(y):
                    keypoints[ind[0]].append([x, y])
                break

    # keypoints = []
    # for ai, ann in enumerate(mpii_annorect):
    #     if ai == ind_to_exclude:
    #         continue
    #
    #     # add keypoint without visibility element (not needed in 'other_keypoints')
    #     keypoints.append(get_movenet_keypoints(ann, h_rescaling_factor, w_rescaling_factor, add_visibility=False))

    return keypoints


def _is_coord_inside_boundaries(val):
    if 0 < val < 1:
        return True
    return False


def _mpii_to_movenet(poses_mpii, output_folder, image_name, img, crop_from_bbox=False):

    poses_movenet = []
    for ai, p_mpii in enumerate(poses_mpii):

        try:
            p_movenet = dict()

            h, w = img.shape

            # if the image is going to be cropped around the pose, then
            # all keypoints must be transformed to the bbox coord. system
            if crop_from_bbox:

                # assign image name
                elems = image_name.split('.')
                ext = elems[-1]
                name = '.'.join(elems[:-1])
                img_cropped_name = f'{name}_{ai}.{ext}'
                p_movenet['img_name'] = img_cropped_name

                bbox = _compute_bbox(p_mpii)

                # compute padding
                pad_top = 0
                pad_left = 0
                pad_right = 0
                pad_bottom = 0
                if bbox[0] < 0:
                    pad_left = -bbox[0] + 1
                if bbox[1] < 0:
                    pad_top = -bbox[1] + 1
                if bbox[2] > w:
                    pad_right = bbox[2] - w + 1
                if bbox[3] > h:
                    pad_bottom = bbox[3] - h + 1

                # adjust bbox coordinates with padding
                bbox[0] += pad_left
                bbox[1] += pad_top
                bbox[2] += pad_left
                bbox[3] += pad_top

                # compute coordinates rescaling factor
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                w_rescaling_factor = 1 / bbox_w
                h_rescaling_factor = 1 / bbox_h

                # compute coordinates x and y shift
                x_shift = pad_left - bbox[0]
                y_shift = pad_top - bbox[1]

                # p_movenet['bbox_unscaled'] = bbox

                # make a new padded image
                pad_img = np.zeros((h + pad_top + pad_bottom, w + pad_left + pad_right))
                pad_img[pad_top:pad_top + h, pad_left:pad_left + w] = img

                # crop bbox and save
                img_cropped = pad_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                file_path = output_folder / img_cropped_name
                if not cv2.imwrite(str(file_path), img_cropped):
                    print(f'could not save image to {str(file_path)}')
            else:

                p_movenet['img_name'] = image_name

                # compute coordinates rescaling factor
                h_rescaling_factor = 1 / h
                w_rescaling_factor = 1 / w

                # compute coordinates x and y shift
                x_shift = 0
                y_shift = 0

            p_movenet['keypoints'] = _get_movenet_keypoints(p_mpii, h_rescaling_factor, w_rescaling_factor, x_shift, y_shift)

            # TODO: movenet discards poses with less than 8 keypoints; keep the check?
            # num_keypoints = len(p_movenet['keypoints'])
            # if num_keypoints < 8:
            #     print(f'Not enough keypoints ({num_keypoints}) for pose {ai} in image {str(img_path.resolve())}')
            #     continue

            p_movenet['head_size'] = _compute_head_size(p_mpii)
            p_movenet['head_size_scaled'] = _compute_head_size(p_mpii, h_rescaling_factor, w_rescaling_factor)
            p_movenet['center'] = [(p_mpii['objpos']['x'] + x_shift) * w_rescaling_factor,
                                   (p_mpii['objpos']['y'] + y_shift) * h_rescaling_factor]
            p_movenet['other_centers'] = _get_movenet_other_centers(ai, poses_mpii, h_rescaling_factor, w_rescaling_factor, x_shift, y_shift)
            p_movenet['other_keypoints'] = _get_movenet_other_keypoints(ai, poses_mpii, h_rescaling_factor, w_rescaling_factor, x_shift, y_shift)
        except:
            print(f'could not process pose {ai} for image {image_name}')
            continue

        poses_movenet.append(p_movenet)

    return poses_movenet


def export_to_movenet(data_ann: dict, image_folder: pathlib.Path, output_folder: pathlib.Path, crop_poses: bool = True) -> None:

    """Export MPII images and annotations to Movenet's compatible format.

    The function converts MPII annotations to Movenet's compatible json. Example format


    If crop_poses is set to True, then frames are cropped using the bounding boxes around single poses and saved to the
    output folder as <frame_name>_<pose_num>.<ext>; additionally, keypoints coordinates are transformed to the bounding
    box reference system.

    Parameters
    ----------
    data_ann: dict
        Dictionary containing MPII annotation data
    image_folder: pathlib.Path
        Path to the images folder
    output_folder: pathlib.Path
        Path to the output folder where the TOS-like frames and the json file will be saved
    crop_poses: bool
        flag indicating if frames must be cropped around single poses (default: True)
    """

    iterator = mpii_parse.MPIIIterator(data_ann, image_folder)

    poses = []
    for fi, (_, poses_ann, img_name) in enumerate(tqdm(iterator)):

        file_path = image_folder / img_name
        file_path = str(file_path.resolve())
        img = cv2.imread(file_path)

        if img is None:
            print(f'image {file_path} does not exist')
            continue

        movenet_poses = _mpii_to_movenet(poses_ann, output_folder, img_name, img, crop_from_bbox=crop_poses)

        if len(movenet_poses) == 0:
            print(f'no annotations for image {file_path}')
            continue

        poses.extend(movenet_poses)

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

    export_to_movenet(data_ann, img_folder_path, output_folder_path, crop_poses=args.crop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ################
    # general params
    ################

    parser.add_argument('-a', help='path to annotations file')
    parser.add_argument('-i', help='path to images folder')
    parser.add_argument('-o', help='path to output folder')

    ###################
    # annotation params
    ###################

    parser.add_argument('-crop', help='flag; if specified, images will be cropped around single poses and saved, '
                                      'keypoint coordinates will be transformed to the bbox coordinates space',
                        dest='crop', action='store_true')
    parser.set_defaults(crop=True)

    args = parser.parse_args()

    main(args)
