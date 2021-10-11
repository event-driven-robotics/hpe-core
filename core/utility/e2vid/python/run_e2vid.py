
import argparse
import torch

from image_reconstructor import ImageReconstructor
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.loading_utils import load_model, get_device
from utils.timers import Timer


# global variables
args = None
compute_voxel_grid_on_cpu = False
device = None
model = None
reconstructor = None
start_index = 0
width = 0
height = 0


def init_model(sensor_height, sensor_width, window_size, events_per_pixel):

    parser = argparse.ArgumentParser()

    parser.add_argument('-sw', '--sensor_width', default=sensor_width, type=int, help='width of the sensor')
    parser.add_argument('-sh', '--sensor_height', default=sensor_height, type=int, help='height of the sensor')
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('-N', '--window_size', default=window_size, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=events_per_pixel, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    parser.add_argument('-o', '--output_folder', default=None, type=str)  # if None, will not write the images to disk
    parser.add_argument('--dataset_name', default='reconstruction', type=str)

    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=True)

    """ Display """
    parser.add_argument('--display', dest='display', action='store_true')
    parser.set_defaults(display=False)

    parser.add_argument('--show_events', dest='show_events', action='store_true')
    parser.set_defaults(show_events=False)

    parser.add_argument('--event_display_mode', default='red-blue', type=str,
                        help="Event display mode ('red-blue' or 'grayscale')")

    parser.add_argument('--num_bins_to_show', default=-1, type=int,
                        help="Number of bins of the voxel grid to show when displaying events (-1 means show all the bins).")

    parser.add_argument('--display_border_crop', default=0, type=int,
                        help="Remove the outer border of size display_border_crop before displaying image.")

    parser.add_argument('--display_wait_time', default=1, type=int,
                        help="Time to wait after each call to cv2.imshow, in milliseconds (default: 1)")

    """ Post-processing / filtering """

    # (optional) path to a text file containing the locations of hot pixels to ignore
    parser.add_argument('--hot_pixels_file', default=None, type=str)

    # (optional) unsharp mask
    parser.add_argument('--unsharp_mask_amount', default=0.3, type=float)
    parser.add_argument('--unsharp_mask_sigma', default=1.0, type=float)

    # (optional) bilateral filter
    parser.add_argument('--bilateral_filter_sigma', default=0.0, type=float)

    # (optional) flip the event tensors vertically
    parser.add_argument('--flip', dest='flip', action='store_true')
    parser.set_defaults(flip=False)

    """ Tone mapping (i.e. rescaling of the image intensities)"""
    parser.add_argument('--Imin', default=0.0, type=float,
                        help="Min intensity for intensity rescaling (linear tone mapping).")
    parser.add_argument('--Imax', default=1.0, type=float,
                        help="Max intensity value for intensity rescaling (linear tone mapping).")
    parser.add_argument('--auto_hdr', dest='auto_hdr', action='store_true',
                        help="If True, will compute Imin and Imax automatically.")
    parser.set_defaults(auto_hdr=False)
    parser.add_argument('--auto_hdr_median_filter_size', default=10, type=int,
                        help="Size of the median filter window used to smooth temporally Imin and Imax")

    """ Perform color reconstruction? (only use this flag with the DAVIS346color) """
    parser.add_argument('--color', dest='color', action='store_true')
    parser.set_defaults(color=False)

    """ Advanced parameters """
    # disable normalization of input event tensors (saves a bit of time, but may produce slightly worse results)
    parser.add_argument('--no-normalize', dest='no_normalize', action='store_true')
    parser.set_defaults(no_normalize=False)

    # disable recurrent connection (will severely degrade the results; for testing purposes only)
    parser.add_argument('--no-recurrent', dest='no_recurrent', action='store_true')
    parser.set_defaults(no_recurrent=False)

    global args
    args = parser.parse_args()

    global width
    width = args.sensor_width
    global height
    height = args.sensor_height
    print('Sensor size: {} x {}'.format(width, height))

    global model
    model = load_model('pretrained/E2VID_lightweight.pth.tar')  # TODO: specify correct path for c++ wrapper

    global device
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    global reconstructor
    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

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

    global start_index
    start_index = initial_offset + sub_offset

    global compute_voxel_grid_on_cpu
    compute_voxel_grid_on_cpu = args.compute_voxel_grid_on_cpu
    if compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')

    return True


def predict_grayscale_frame(event_window):

    global start_index

    last_timestamp = event_window[-1, 0]

    with Timer('Building event tensor'):
        if compute_voxel_grid_on_cpu:
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

    # TODO: return predicted image!!!
