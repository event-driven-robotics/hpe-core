import sys
import yarp
import numpy as np
import cv2
import os

HOME_DIR = '/usr/local/code'
sys.path.append(os.path.join(HOME_DIR, 'gl_hpe/'))
import experimenting
import torch
from experimenting.utils.visualization import plot_skeleton_2d, plot_skeleton_3d, plot_skeleton_2d_lined
from experimenting.utils.skeleton_helpers import Skeleton
from experimenting.utils import utilities
from experimenting import utils

class GlHpeModule(yarp.RFModule):

    def __init__(self):
        yarp.RFModule.__init__(self)
        self.input_port = yarp.BufferedPortImageMono()
        self.output_port = yarp.BufferedPortImageMono()
        self.stamp = yarp.Stamp()
        self.counter = 0
        self.image_w = 400 # Size of image expected from the framer.
        self.image_h = 300 #
        # self.np_input = None
        self.yarp_image = yarp.ImageMono()
        self.yarp_image_out = yarp.ImageRgb()

        self.datadir = HOME_DIR
        self.checkpoint_path = os.path.join(HOME_DIR, "gl_hpe/checkpoint")
        self.P_mat_dir = os.path.join(self.datadir, 'gl_hpe/P_matrices/')

        self.ch_idx = 3
        self.resultsPath = os.path.join(self.datadir, 'outputs')
        self.image_w_model = 346 # Size of the image expected by the model
        self.image_h_model = 260 #
        self.output_w = 640  # Size of the image expected by yarp
        self.output_h = 480  #
        self.fname = None
        self.fname_ts = None
        self.last_timestamp = 0.0
        # self.model = None
        # self.read_image = None

    def configure(self, rf):
        # Initialise YARP
        yarp.Network.init()
        if not yarp.Network.checkNetwork(2):
            print("Could not find network! Run yarpserver and try again.")
            exit(-1)

        # set the module name used to name ports
        self.setName((rf.check("name", yarp.Value("/glhpeModule")).asString()))

        # set the output file name
        self.fname = rf.check("write_sk", yarp.Value("pred_2D.npy")).asString()
        self.fname_ts = rf.check("write_ts", yarp.Value("pred_ts.npy")).asString()

        # open io ports
        if not self.input_port.open(self.getName() + "/img:i"):
            print("Could not open input port")
            return False
        if not self.output_port.open(self.getName() + "/img:o"):
            print("Could not open output port")
            return False

        # read flags and parameters
        self.dhpcore = experimenting.dataset.DHP19Core('test',
                                                       data_dir=os.path.join(self.datadir,
                                                                             'time_count_dataset/movements_per_frame'),
                                                       joints_dir=os.path.join(self.datadir,
                                                                               "time_count_dataset/labels_full_joints/"),
                                                       hm_dir="", labels_dir="", preload_dir="", n_joints=13,
                                                       n_classes=33,
                                                       partition='cross-subject', n_channels=1, cams=[1, 3],
                                                       movements=None, test_subjects=[6, 7])

        # example_flag = rf.check("example_flag") and rf.check("example_flag", yarp.Value(True)).asBool()
        # default_value = 0.1
        # example_parameter = rf.check("example_parameter", yarp.Value(default_value)).asDouble()
        #
        # # do any other set-up required here
        self.model = utilities.load_model(self.checkpoint_path, "MargiposeEstimator",
                                          core=self.dhpcore).eval().double()
        if self.ch_idx == 0:
            self.P_mat_cam = np.load(os.path.join(self.P_mat_dir, 'P1.npy'))
        elif self.ch_idx == 3:
            self.P_mat_cam = np.load(os.path.join(self.P_mat_dir, 'P2.npy'))
        elif self.ch_idx == 2:
            self.P_mat_cam = np.load(os.path.join(self.P_mat_dir, 'P3.npy'))
        elif self.ch_idx == 1:
            self.P_mat_cam = np.load(os.path.join(self.P_mat_dir, 'P4.npy'))
        self.extrinsics_matrix, self.camera_matrix = utils.decompose_projection_matrix(self.P_mat_cam)

        return True

    def getPeriod(self):
        return 0  # period of synchronous thread, return 0 update module called as fast as it can

    def interruptModule(self):
        # interrupting all the ports
        self.input_port.interrupt()
        return True

    def close(self):
        # closing ports
        self.input_port.close()
        cv2.destroyAllWindows()
        return True

    def updateModule(self):
        # synchronous update called every get period seconds.
        print("Press space at the image window to end the program.")

        # Preparing input and output image buffers
        np_input = np.ones((self.image_h, self.image_w), dtype=np.uint8)
        self.yarp_image.resize(self.image_w, self.image_h)
        self.yarp_image.setExternal(np_input.data, np_input.shape[1], np_input.shape[0])

        # np_output = np.ones((self.output_h, self.output_w, 3), dtype=np.uint8)
        # self.yarp_image_out.resize(self.output_w, self.output_h)
        # self.yarp_image_out.setExternal(np_output.data, np_output.shape[1], np_output.shape[0])

        # Read the image
        read_image = self.input_port.read()
        self.input_port.getEnvelope(self.stamp)
        stamp = self.stamp.getTime()
        # print(self.stamp.getTime())

        # End of input file
        if self.last_timestamp == stamp:
            return False

        self.counter += 1  # can be used to interrupt the program
        self.yarp_image.copy(read_image)
        input_image = np.copy(np_input[:self.image_h_model, :self.image_w_model]) / 255.0
        if len(input_image.shape) == 2:
            input_image = np.expand_dims(input_image, -1)
            input_image = np.expand_dims(input_image, 0)

        # Predict the pose
        torch_image = torch.from_numpy(input_image)
        preds, outs = self.model(torch_image.permute(0, -1, 1, 2))
        pred_sk = Skeleton(preds[0].detach().numpy()).denormalize(260, 346,
                                                                  camera=torch.tensor(self.camera_matrix)). \
            reproject_onto_world(torch.tensor(self.extrinsics_matrix))
        pred_joints = pred_sk.get_2d_points(260, 346, p_mat=torch.tensor(self.P_mat_cam))

        # Obtain the 2D prediction as an image
        fig2D = plot_skeleton_2d(input_image[0].squeeze(), pred_joints, return_figure=True, lines=True)
        fig2D.canvas.draw()
        img = np.fromstring(fig2D.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig2D.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Visualize the result
        cv2.imshow("output", img)
        print(img.shape)
        k = cv2.waitKey(10)
        # cv2.imwrite(os.path.join(self.resultsPath, 'input_'+str(self.counter), '.png'), input_image)
        # cv2.imwrite(os.path.join(self.resultsPath, 'output_2D_'+str(self.counter),'.png'), img)

        # write the Results into numpy arrays
        # utilities.save_2D_prediction(pred_joints, fname=os.path.join(self.resultsPath, self.fname), overwrite=False)
        # utilities.save_timestamp(stamp, fname=os.path.join(self.resultsPath, self.fname_ts), overwrite=False)

        #        np_output[:, :] = img
        # self.output_port.write(self.yarp_image_out)
        self.last_timestamp = stamp
        if k == 32:
            return False
        return True


if __name__ == '__main__':
    # prepare and configure the resource finder
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("event-driven")
    # rf.setDefaultConfigFile("exampleModule.ini")
    rf.configure(sys.argv)

    # create the module
    module = GlHpeModule()
    module.runModule(rf)
