
import argparse
import os
import stat
import torch

from image_reconstructor import ImageReconstructor
from mat_files import Dhp19EventsIterator, loadmat
from options.inference_options import set_inference_options
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.loading_utils import load_model, get_device
from utils.timers import Timer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-sw', '--sensor_width', default=346, type=int, help='width of the sensor')
    parser.add_argument('-sh', '--sensor_height', default=260, type=int, help='height of the sensor')
    parser.add_argument('-cid', '--cam_id', default=0, type=int, help='id of the DVS camera')
    parser.add_argument('-ie', '--events_file', required=True, type=str)
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('-N', '--window_size', default=7500, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()

    # for every channel
    #     for every event window
    #         run e2vid
    #         save frame
    #         (save pose? check if they are already available)
    #         (save frames and poses in .h5 files?)

    width = args.sensor_width
    height = args.sensor_height
    print('Sensor size: {} x {}'.format(width, height))

    # create subfolder for the selected cam
    subfolder_path = f'{args.output_folder}/{args.cam_id}'
    args.output_folder = subfolder_path
    try:
        os.umask(0)
        os.makedirs(subfolder_path)
    except FileExistsError:
        print(f'Folder {subfolder_path} already exists')
        exit(0)
    except:
        print(f'Could not create folder {subfolder_path}')
        exit(0)

    # Load model
    model = load_model('pretrained/E2VID_lightweight.pth.tar')
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    """ Read chunks of events using Pandas """

    # Loop through the events and reconstruct images
    N = args.window_size
    if not args.fixed_duration:
        if N is None:
            N = int(width * height * args.num_events_per_pixel)
            print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
                N, args.num_events_per_pixel))
        else:
            print('Will use {} events per tensor (user-specified)'.format(N))
            mean_num_events_per_pixel = float(N) / float(width * height)
            if mean_num_events_per_pixel < 0.1:
                print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
            elif mean_num_events_per_pixel > 1.5:
                print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset

    if args.compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')

    # if args.fixed_duration:
    #     event_window_iterator = FixedDurationEventReader(args.events_file,
    #                                                      duration_ms=args.window_duration,
    #                                                      start_index=start_index)
    # else:
    #     event_window_iterator = FixedSizeEventReader(args.events_file, num_events=N, start_index=start_index)

    data_events = loadmat(args.events_file)

    event_window_iterator = Dhp19EventsIterator(data=data_events, cam_id=args.cam_id, window_size=7500)

    with Timer('Processing entire dataset'):
        for event_window in event_window_iterator:

            last_timestamp = event_window[-1, 0]

            with Timer('Building event tensor'):
                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(event_window,
                                                        num_bins=model.num_bins,
                                                        width=width,
                                                        height=height)
                    event_tensor = torch.from_numpy(event_tensor)
                else:
                    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=model.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)

            num_events_in_window = event_window.shape[0]
            reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

            start_index += num_events_in_window
