
import argparse
import numpy as np
import os

from utils import mat_files
from evaluation.utils import DHP19_BODY_PARTS


def extract_2d_poses(data_events, data_vicon, projection_mat, camera_id, window_size):

    avg_poses_3d = extract_3d_poses(data_events, data_vicon, camera_id, window_size)

    avg_poses_2d = project_poses_to_2d(avg_poses_3d, np.transpose(projection_mat))
    # avg_poses_2d = project_poses_to_2d_old(np.transpose(avg_poses_3d[0]), projection_mat)

    # is this needed?
    # avg_poses_2d = avg_poses_2d.astype(np.uint16)

    return avg_poses_2d


def extract_3d_poses(data_events, data_vicon, camera_id, window_size):

    # # normalize events timestamps
    start_time = data_events['out']['extra']['startTime']
    # data_events['out']['data'][f'cam{camera_id}']['dvs']['ts'] = (data_events['out']['data'][f'cam{camera_id}']['dvs'][
    #                                                                'ts'] - start_time) * 1e-6
    #
    # # create an array of timestamps for the vicon's 3d poses
    # dt = 10000  # time step
    # poses_timestamps = np.arange(data_events['out']['extra']['ts'][0] - start_time,
    #                              data_events['out']['extra']['ts'][-1] - start_time + dt,
    #                              dt) * 1e-6  # Vicon timestams @ 100Hz
    # diff = len(poses_timestamps) - data_vicon['XYZPOS']['head'].shape[0]
    # if diff > 0:
    #     poses_timestamps = poses_timestamps[:-diff]

    ##################################################################
    # for every window of events, compute the corresponding 2d pose by
    # - averaging all the 3d poses with matching timestamps
    # - projecting the average 3d pose to 2d
    ##################################################################

    # compute average 3d poses
    event_window_iterator = mat_files.Dhp19EventsIterator(data=data_events, cam_id=camera_id, window_size=window_size)
    avg_poses_3d = np.zeros(shape=(len(event_window_iterator), len(DHP19_BODY_PARTS), 3))
    for wi, event_window in enumerate(event_window_iterator):

        # TODO: what if event_window is empty?

        # get all 3d poses that fall into the events window
        poses_start_ind = int(np.floor((event_window[0, 0] - start_time) * 1e-4))  # + 1
        poses_end_ind = int(np.floor((event_window[-1, 0] - start_time) * 1e-4))  # + 1
        # TODO: what if there are no poses?
        # find closest ones...

        # compute the average 3d pose
        for body_part in DHP19_BODY_PARTS:
            coords = data_vicon['XYZPOS'][body_part][poses_start_ind:poses_end_ind, :]
            avg_poses_3d[wi, DHP19_BODY_PARTS[body_part], :] = np.nanmean(coords, axis=0)

    return avg_poses_3d


def project_poses_to_2d(poses_3d, projection_mat):

    # use homogeneous coordinates representation to project 3d XYZ coordinates to 2d UV pixel coordinates
    vicon_xyz_homog = np.concatenate([poses_3d, np.ones([len(poses_3d), 13, 1])], axis=2)
    coord_pix_homog = np.matmul(vicon_xyz_homog, projection_mat)
    coord_pix_homog_norm = coord_pix_homog / np.reshape(coord_pix_homog[:, :, -1], (len(poses_3d), 13, 1))

    u = coord_pix_homog_norm[:, :, 0]
    v = mat_files.DHP19_SENSOR_HEIGHT - coord_pix_homog_norm[:, :, 1]  # flip v coordinate to match the image direction

    # mask is used to make sure that pixel positions are in frame range.
    mask = np.ones(u.shape).astype(np.float32)
    mask[np.isnan(u)] = 0
    mask[np.isnan(v)] = 0
    mask[u > mat_files.DHP19_SENSOR_WIDTH] = 0
    mask[u <= 0] = 0
    mask[v > mat_files.DHP19_SENSOR_HEIGHT] = 0
    mask[v <= 0] = 0

    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    return np.stack((v, u), axis=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--events_file_path', help='', required=True)
    parser.add_argument('-v', '--vicon_file_path', help='', required=True)
    parser.add_argument('-c', '--camera_id', help='', required=True, type=int)
    parser.add_argument('-p', '--projection_matrix_file_path', help='', required=True)
    parser.add_argument('-w', '--window_size', help='', default=mat_files.DHP19_CAM_FRAME_EVENTS_NUM)
    parser.add_argument('-o', '--output_folder', help='', required=True)
    parser.add_argument('-td', '--two_dimensional', dest='two_dimensional', help='', action='store_true')
    parser.set_defaults(two_dimensional=True)
    args = parser.parse_args()

    # read data from .mat files
    data_events = mat_files.loadmat(args.events_file_path)
    data_vicon = mat_files.loadmat(args.vicon_file_path)
    proj_mat = np.load(args.projection_matrix_file_path)

    if args.two_dimensional:
        poses = extract_2d_poses(data_events, data_vicon, proj_mat, args.camera_id, args.window_size)
        file_name = f'2d_poses_cam_{args.camera_id}_{args.window_size}_events.npy'
    else:
        poses = extract_3d_poses(data_events, data_vicon, args.camera_id, args.window_size)
        file_name = f'3d_poses_cam_{args.camera_id}_{args.window_size}_events.npy'

    np.save(os.path.join(args.output_folder, file_name), poses, allow_pickle=False)
