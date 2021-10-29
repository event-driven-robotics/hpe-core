
import os
import torch

import e2vid_config

from image_reconstructor import ImageReconstructor
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.loading_utils import load_model, get_device
from utils.timers import Timer


# global variables
model = None


def init_model():
    global model
    model = E2Vid(e2vid_config)


def predict_grayscale_frame(event_window):
    global model
    return model.predict_grayscale_frame(event_window)


class E2Vid:

    def __init__(self, options):

        self.width = options.sensor_width
        self.height = options.sensor_height
        print('Sensor size: {} x {}'.format(self.width, self.height))

        self.device = get_device(options.use_gpu)

        self.model = load_model(os.path.join(os.getenv('E2VID_PYTHON_DIR'), 'pretrained/E2VID_lightweight.pth.tar'))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.reconstructor = ImageReconstructor(self.model, self.height, self.width, self.model.num_bins, options)

        N = options.window_size
        if not options.fixed_duration:
            if N is None:
                N = int(self.width * self.height * options.num_events_per_pixel)
                print(
                    'Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
                        N, options.num_events_per_pixel))
            else:
                print('Will use {} events per tensor (user-specified)'.format(N))
                mean_num_events_per_pixel = float(N) / float(self.width * self.height)
                if mean_num_events_per_pixel < 0.1:
                    print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                            The reconstruction results might be suboptimal.'.format(N))
                elif mean_num_events_per_pixel > 1.5:
                    print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                            The reconstruction results might be suboptimal.'.format(N))

        initial_offset = options.skipevents
        sub_offset = options.suboffset

        self.start_index = initial_offset + sub_offset

        self.compute_voxel_grid_on_cpu = options.compute_voxel_grid_on_cpu
        if self.compute_voxel_grid_on_cpu:
            print('Will compute voxel grid on CPU.')

        print('------------------------------------------------------------------\n')
        print('model initialized')

        # TODO: PRINT THREAD ID, IN PYTHON AND C++!!!!!!!!!!!!!!!!!!
        # TODO: TRY TO INIT E2VID IN MAIN!!!!!!!!!!!!!!!!!!

    def model_share_memory(self):
        self.model.share_memory()

    def predict_grayscale_frame(self, event_window):

        last_timestamp = event_window[-1, 0]

        with Timer('Building event tensor'):
            if self.compute_voxel_grid_on_cpu:
                event_tensor = events_to_voxel_grid(event_window,
                                                    num_bins=self.model.num_bins,
                                                    width=self.width,
                                                    height=self.height)
                event_tensor = torch.from_numpy(event_tensor)
            else:
                event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                            num_bins=self.model.num_bins,
                                                            width=self.width,
                                                            height=self.height,
                                                            device=self.device)

        num_events_in_window = event_window.shape[0]
        grayscale_frame = self.reconstructor.update_reconstruction(event_tensor,
                                                                   self.start_index + num_events_in_window,
                                                                   last_timestamp)

        self.start_index += num_events_in_window

        return grayscale_frame
