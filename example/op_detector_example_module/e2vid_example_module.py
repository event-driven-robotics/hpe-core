
import argparse
import yarp
import sys
import numpy as np
import threading
import cv2

import e2vid_config
from e2vid import E2Vid


class E2VidExampleModule(yarp.RFModule):

    def __init__(self, e2vid_options):
        yarp.RFModule.__init__(self)
        self.image = np.zeros((e2vid_config.sensor_height, e2vid_config.sensor_width), dtype=np.uint8)
        self.image_buf = np.zeros((e2vid_config.sensor_height, e2vid_config.sensor_width), dtype=np.uint8)
        self.events = np.zeros((7500, 4), dtype=np.float64)
        self.input_port = yarp.BufferedPortBottle()

        self.out_buf_image = yarp.ImageMono()
        self.out_buf_image.resize(e2vid_config.sensor_width, e2vid_config.sensor_height)
        self.out_buf_image.setExternal(self.image.data, self.image.shape[1], self.image.shape[0])

        self.output_port = yarp.Port()
        self.rpc_port = yarp.RpcServer()

        cv2.namedWindow("e2vid", cv2.WINDOW_NORMAL)
        # self.mutex = threading.Lock()

        self.e2vid = E2Vid(e2vid_options)

    def configure(self, rf):
        # set the module name used to name ports
        self.setName((rf.check("name", yarp.Value("/e2vid_example_module")).asString()))

        # open io ports
        if not self.input_port.open(self.getName() + "/AE:i"):
            print("Could not open input port")
            return False
        self.input_port.setStrict()

        if not self.output_port.open(self.getName() + "/img:o"):
            print("Could not open output port")
            return False

        if not self.rpc_port.open(self.getName() + "/rpc"):
            print("Could not open rpc port")
            return False
        self.attach_rpc_server(self.rpc_port)  # rpc port receives command in the respond method

        # read flags and parameters
        example_flag = rf.check("example_flag") and rf.check("example_flag", yarp.Value(True)).asBool()
        default_value = 0.1
        example_parameter = rf.check("example_parameter", yarp.Value(default_value)).asDouble()

        # do any other set-up required here
        # start the asynchronous and synchronous threads
        threading.Thread(target=self.run).start()

        return True

    def respond(self, command, reply):
        # Add any command you want to receive from rpc here
        print(command.toString())
        reply.addString('ok')
        return True

    def getPeriod(self):
        return 0.  # period of synchronous thread, return 0 update module called as fast as it can

    def interruptModule(self):
        # interrupting all the ports
        self.input_port.interrupt()
        self.rpc_port.interrupt()
        return True

    def close(self):
        # closing ports
        self.input_port.close()
        self.rpc_port.close()
        cv2.destroyAllWindows()
        return True

    def updateModule(self):
        # synchronous update called every get period seconds.

        # Put visualization, debug prints, etc... here
        if self.image is None:
            print('image is None')
            return True

        self.output_port.write(self.out_buf_image)

        cv2.imshow("e2vid", self.image)
        cv2.waitKey(10)
        return True

    def run(self):

        # asynchronous thread runs as fast as it can
        while not self.isStopping():

            bottle = self.input_port.read()

            # Data in the bottle is organized as <event_type> (<timestamp 1> <event 1> .... <timestamp n> <event n>)
            vType = bottle.get(0).asString()
            if vType != "AE":
                continue
            event_bottle = np.array(bottle.get(1).toString().split(' '), dtype=np.uint32).reshape(-1, 2)

            # get timestamp
            timestamps = event_bottle[:, 0]
            timestamps = timestamps.reshape(-1, 1).astype(np.float32)

            # get x and y coordinates
            events_buf = event_bottle[:, 1]
            y = events_buf >> 12 & 0xFF
            y = y.reshape(-1, 1)
            x = events_buf >> 1 & 0x1FF
            x = x.reshape(-1, 1)

            # get polarity
            pol = events_buf & 0x01
            pol = pol.reshape(-1, 1)

            self.events = np.concatenate((timestamps, x, y, pol), axis=1)
            self.image_buf = self.e2vid.predict_grayscale_frame(self.events)
            # self.mutex.acquire()
            self.image[:, :] = self.image_buf  # self.image is a shared resource between threads
            # self.mutex.release()


if __name__ == '__main__':

    #######################
    # parse e2vid options #
    #######################

    parser = argparse.ArgumentParser()

    parser.add_argument('-sw', '--sensor_width', default=304, type=int, help="Width of the sensor (304 is SIE's sensor width)")
    parser.add_argument('-sh', '--sensor_height', default=240, type=int, help="Height of the sensor (240 is SIE's sensor height)")
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('-N', '--window_size', default=7500, type=int, help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=33.33, type=float, help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float, help='in case N (window size) is not specified, it will be automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)
    parser.add_argument('-out', '--output_folder', default=None, type=str, help="if None, will not write the images to disk")
    parser.add_argument('--gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=True)
    parser.add_argument('-dn', '--dataset_name', default='reconstruction', type=str)

    # display
    parser.add_argument('--display', dest='display', action='store_true')
    parser.set_defaults(display=False)
    parser.add_argument('--show_events', dest='show_events', action='store_true')
    parser.set_defaults(show_events=False)
    parser.add_argument('--event_display_mode', default='red-blue', type=str, help="Event display mode ('red-blue' or 'grayscale')")
    parser.add_argument('--num_bins_to_show', default=-1, type=int, help="Number of bins of the voxel grid to show when displaying events (-1 means show all the bins)")
    parser.add_argument('--display_border_crop', default=0, type=int, help="Remove the outer border of size display_border_crop before displaying image")
    parser.add_argument('--display_wait_time', default=1, type=int, help="Time to wait after each call to cv2.imshow, in milliseconds (default: 1)")

    # post-processing / filtering
    parser.add_argument('--hot_pixels_file', default=None, type=str,
                        help="(optional) path to a text file containing the locations of hot pixels to ignore")
    parser.add_argument('--unsharp_mask_amount', default=0.3, type=float, help='(optional) unsharp mask amount')
    parser.add_argument('--unsharp_mask_sigma', default=1.0, type=float, help='(optional) unsharp mask sigma')
    parser.add_argument('--bilateral_filter_sigma', default=0.0, type=float, help='(optional) bilateral filter')
    parser.add_argument('--flip', dest='flip', action='store_true', help='(optional) flip the event tensors vertically')
    parser.set_defaults(flip=False)

    # tone mapping (i.e. rescaling of the image intensities)
    parser.add_argument('--Imin', default=0.0, type=float, help='Min intensity for intensity rescaling (linear tone mapping)')
    parser.add_argument('--Imax', default=1.0, type=float, help='Max intensity value for intensity rescaling (linear tone mapping)')
    parser.add_argument('--auto_hdr', dest='auto_hdr', action='store_true', help='If True, will compute Imin and Imax automatically')
    parser.set_defaults(auto_hdr=False)
    parser.add_argument('--auto_hdr_median_filter_size', default=10, type=int, help="Size of the median filter window used to smooth temporally Imin and Imax")
    parser.add_argument('--color', dest='color', action='store_true', help='Perform color reconstruction? (only use this flag with the DAVIS346color')
    parser.set_defaults(color=False)

    # advanced parameters
    parser.add_argument('--no_normalize', dest='no_normalize', action='store_true', help='disable normalization of input event tensors (saves a bit of time, but may produce slightly worse results')
    parser.set_defaults(no_normalize=False)
    parser.add_argument('--no_recurrent', dest='no_recurrent', action='store_true', help='disable recurrent connection (will severely degrade the results; for testing purposes only')
    parser.set_defaults(no_recurrent=True)

    e2vid_options = parser.parse_args()

    # initialise YARP
    yarp.Network.init()
    if not yarp.Network.checkNetwork(2):
        print("Could not find network! Run yarpserver and try again.")
        exit(-1)

    # prepare and configure the resource finder
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("eventdriven")
    rf.configure(sys.argv)

    module = E2VidExampleModule(e2vid_options)
    module.runModule(rf)
