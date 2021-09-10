
import argparse
import cv2
import numpy as np
import os

from pathlib import Path

from utils import metrics, plots
from utils import parse_openpose_keypoints_json
from dhp19.utils import DHP19_BODY_PARTS, openpose_to_dhp19


# def evaluate_openpose():
#     pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate OpenPose on DHP19, plotting predicted and ground-truth poses \
                                                 on input frames and computing Mean Per Joint Position Error metric')
    parser.add_argument('-gt', '--ground_truth_path', help='Path to the .npy file containing the ground truth poses for every frame (computed using dhp19/extract_gt_poses.py)', required=True)
    parser.add_argument('-p', '--predictions_folder', help='Path to the folder containing the .json files (one for each frame) with predicted poses (computed using OpenPose)', required=True)
    parser.add_argument('-if', '--images_folder', help='Path to the folder containing the image frames')
    parser.add_argument('-o', '--output_folder', help='Path to the folder where evaluation results will be saved', required=True)
    args = parser.parse_args()

    # read data
    poses_gt = np.load(args.ground_truth_path)
    poses_pred_files = sorted(Path(args.predictions_folder).glob('*.json'))
    image_files = sorted(Path(args.images_folder).glob('*.png'))

    # check if the number of images, gt and predicted poses is the same
    assert len(poses_gt) == len(poses_pred_files)
    assert len(poses_pred_files) == len(image_files)

    # read and preprocess predicted poses
    poses_pred = np.zeros((len(poses_pred_files), len(DHP19_BODY_PARTS), 2), dtype=int)
    for pi, path in enumerate(poses_pred_files):
        op_pose = parse_openpose_keypoints_json(path)
        poses_pred[pi, :] = openpose_to_dhp19(op_pose)

    os.makedirs(args.output_folder, exist_ok=True)

    # compute metrics
    mpjpe = metrics.compute_mpjpe(poses_pred, poses_gt)
    print(f'MPJPE for OpenPose')
    print(f'ground-truth path: {args.ground_truth_path}')
    print(f'predictions folder: {args.predictions_folder}')
    print(f'images folder: {args.images_folder}')
    metrics.print_mpjpe(mpjpe, list(DHP19_BODY_PARTS.keys()))

    # TODO: plot some examples (good and bad based on metrics?)

    # plot predictions and gt
    for i, path in enumerate(image_files):
        img = cv2.imread(str(path))
        plots.plot_poses(img, poses_gt[i], poses_pred[i])
        cv2.imwrite(os.path.join(args.output_folder, path.name), img)
