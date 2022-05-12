
import argparse
import cv2
import os
import pathlib
import shutil

# from bimvee.importIitYarp import importIitYarp
# from tqdm import tqdm

from datasets.utils import mat_files as mat_utils
from datasets.utils.events_representation import EROS

# TODO: add path to v2e folder to PYTHONPATH

def main(args):

    # read image-video mapping
    video_keyframe_mapping_path = pathlib.Path(args.m)
    video_keyframe_mapping_path = pathlib.Path(video_keyframe_mapping_path.resolve())
    video_keyframe_mapping = mat_utils.loadmat(str(video_keyframe_mapping_path))

    # read annotations
    data_ann_path = pathlib.Path(args.a)
    data_ann_path = pathlib.Path(data_ann_path.resolve())
    data_ann = mat_utils.loadmat(str(data_ann_path))

    # create output folders
    output_folder_path = pathlib.Path(args.o)
    output_folder_path = pathlib.Path(output_folder_path.resolve())
    output_folder_path.mkdir(exist_ok=True)

    # get training annotations indices
    train_ann_indices = data_ann['RELEASE']['img_train'] == 1
    train_video_keyframe_mapping = video_keyframe_mapping['annolist_keyframes'][train_ann_indices]

    # ...
    video_frames_batches_path = pathlib.Path(args.f)
    video_frames_batches_path = pathlib.Path(video_frames_batches_path.resolve())

    for keyframe_dict in train_video_keyframe_mapping:
        path_cmps = keyframe_dict['image']['name'].split('/')
        video_id = path_cmps[0]
        keyframe_name = path_cmps[1]

        print(f'************************** processing video {video_id}')

        # create output folders
        output_video_folder_path = output_folder_path / video_id
        try:
            output_video_folder_path.mkdir(parents=True, exist_ok=False)
        except:
            print(f'skipping video {video_id} (directory {str(output_video_folder_path)} already exists)')
            continue

        output_events_folder_path = output_video_folder_path / 'events'
        output_events_folder_path.mkdir(parents=True)
        output_frames_folder_path = output_video_folder_path / 'frames'
        output_frames_folder_path.mkdir(parents=True)

        # get sorted frames names
        mpii_frames_path = video_frames_batches_path / video_id
        mpii_frames_names = [str(fpath.name) for fpath in mpii_frames_path.glob('*.jpg')]
        mpii_frames_names.sort()

        # copy the annotated frame and the n preceding ones to the output folder
        keyframe_ind = mpii_frames_names.index(keyframe_name)
        if args.n <= keyframe_ind:
            start_ind = args.n
        else:
            start_ind = 0
        for fi in range(start_ind, keyframe_ind + 1):
            name = mpii_frames_names[fi]
            source_path_str = str(mpii_frames_path / name)
            target_path_str = str(output_frames_folder_path / name)
            shutil.copy(src=source_path_str, dst=target_path_str)

        # run v2e
        cmd = f'python ../utils/externals/v2e/v2e.py -i {str(output_frames_folder_path)} ' \
              f'--output_folder {str(output_events_folder_path)} ' \
              f'--input_frame_rate 17.0 --dvs{args.v2e_dvs_res} ' \
              f'--no_preview --dvs_aedat2 None --batch_size {args.v2e_batch_size} ' \
              f'--timestamp_resolution=0.001 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 ' \
              f'--pos_thres=0.15 --neg_thres=0.15 --sigma_thres=0.03 --dvs_h5 events.h5 ' \
              f'--cutoff_hz=15 ' \
              f'--skip_video_output'
        os.system(cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='path to file mpii_human_pose_v1_sequences_keyframes.mat')
    parser.add_argument('-a', help='path to the MPII\'s annotation file')
    parser.add_argument('-f', help='path to the MPII\'s video frames batches')
    parser.add_argument('-o', help='path to the output folder')
    parser.add_argument('-n', help='number of frames preceding the annotated one that must be used by v2e', type=int, default=10)
    parser.add_argument('-v2e_dvs_res', help='resolution of v2e output', type=int, default=640)
    parser.add_argument('-v2e_batch_size', help='v2e batch size', type=int, default=6)

    args = parser.parse_args()
    main(args)
