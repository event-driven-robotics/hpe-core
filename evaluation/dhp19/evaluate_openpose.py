
import argparse
import numpy as np

from pathlib import Path

from evaluation.utils import hpe_metrics as metrics


# def evaluate_openpose():
#     pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--ground_truth_path', help='', required=True)
    parser.add_argument('-op', '--openpose_folder', help='', required=True)
    parser.add_argument('-if', '--image_frames', help='')
    parser.add_argument('-o', '--output_folder', help='', required=True)
    args = parser.parse_args()

    # read gt poses from folder
    poses_gt = np.load(args.ground_truth_path)

    # read openpose poses from folder
    dir = Path(args.openpose_folder)
    poses_pred_files = sorted(dir.glob('*.json'))
    poses_pred = np.zeros((len(poses_pred_files), len(metrics.DHP19_BODY_PARTS), 2))

    assert len(poses_gt) == len(poses_pred)

    for pi, path in enumerate(poses_pred_files):
        op_pose = metrics.parse_openpose_keypoints_json(path)
        poses_pred[pi, :] = metrics.openpose_to_dhp19(op_pose)

    # compute metrics
    mpjpe = metrics.compute_mpjpe(poses_pred, poses_gt)

    # plot some examples (good and bad based on metrics?)
    # - needs folder with image frames


