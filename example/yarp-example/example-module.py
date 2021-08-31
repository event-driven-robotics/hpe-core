import yarp
import sys
import numpy as np
import cv2

import experimenting
import event_library as el
import torch
from os.path import join
from experimenting.utils.visualization import plot_skeleton_2d, plot_skeleton_3d
from experimenting.utils.skeleton_helpers import Skeleton
from experimenting.dataset.factory import Joints3DConstructor, BaseDataFactory, SimpleReadConstructor, \
    MinimalConstructor
from experimenting.utils import utilities
from experimenting import utils
import matplotlib.pyplot as plt




# setting the directories and variables
hw = el.utils.get_hw_property('dvs')
datadir = "/media/ggoyal/Shared/data/dhp19_sample/"
dhpcore = experimenting.dataset.DHP19Core('test', data_dir=join(datadir,'time_count_dataset/movements_per_frame'), joints_dir=join(datadir,"time_count_dataset/labels_full_joints/"), hm_dir="",  labels_dir="", preload_dir="", n_joints=13, n_classes=33, partition='cross-subject', n_channels=1, cams=[1, 3], movements=None, test_subjects=[6, 7])
P_mat_dir = join(datadir, 'P_matrices/')
checkpoint_path = "/media/ggoyal/Shared/data/checkpoint_dhp19"
resultsPath = join(datadir, 'outputs/')

ch_idx = 3 # Depends on the input camera!


class ExampleModule(yarp.RFModule):

    def __init__(self):
        yarp.RFModule.__init__(self)
        self.image = np.zeros((344, 260)) # Change???
        # self.image_buf = np.zeros((344, 260)) # Change??????? 344x260
        self.input_port = yarp.BufferedPortImage(Mono)
# alternative buffered port classes depending on the underlying data type:
# - yarp.BufferedPortImage(Rgb|Rgba|Mono|Mono16|Int|Float|RgbFloat)
        cv2.namedWindow("events", cv2.WINDOW_NORMAL)

    def configure(self, rf):

            # Initialise YARP
        yarp.Network.init()
        if not yarp.Network.checkNetwork(2):
            print("Could not find network! Run yarpserver and try again.")
            exit(-1)

        # set the module name used to name ports
        self.setName((rf.check("name", yarp.Value("/exampleModule")).asString()))


        # open io ports
        if not self.input_port.open(self.getName() + "/yarp-example/AE:o"):
            print("Could not open input port")
            return False
        self.input_port.setStrict()

        # read flags and parameters
        example_flag = rf.check("example_flag") and rf.check("example_flag", yarp.Value(True)).asBool()
        default_value = 0.1
        example_parameter = rf.check("example_parameter", yarp.Value(default_value)).asDouble()

        # do any other set-up required here
        self.model = utilities.load_model(checkpoint_path, "MargiposeEstimator", core=dhpcore).eval().double()
        if ch_idx==0: self.P_mat_cam = np.load(join(P_mat_dir,'P1.npy'))
        elif ch_idx==3: self.P_mat_cam = np.load(join(P_mat_dir,'P2.npy'))
        elif ch_idx==2: self.P_mat_cam = np.load(join(P_mat_dir,'P3.npy'))
        elif ch_idx==1: self.P_mat_cam = np.load(join(P_mat_dir,'P4.npy'))
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

        yarp.image = self.input_port.read()
        b_x = np.array(yarp.image)
        preds, outs = self.model(b_x.permute(0, -1, 1, 2))
        pred_sk = Skeleton(preds[0].detach().numpy()).denormalize(260, 346, camera=torch.tensor(self.camera_matrix)).reproject_onto_world(torch.tensor(self.extrinsics_matrix))
        # TODO: PLOT pred_sk in rt?

        plt.figure()
        plt.imshow(self.yarp, cmap='gray')
        plt.plot(y_2d[:,0], y_2d[:,1], '.', c='red', label='gt')

        # %% 3D
        from mpl_toolkits.mplot3d import Axes3D

        x = Sklt[:,0]
        y = Sklt[:,1]
        z = Sklt[:,2]
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x, y, z, zdir='z', s=20, c='red', marker='o', depthshade=True)
        lines_skeleton=skeleton(x,y,z)
        for l in range(len(lines_skeleton)):
            ax.plot(lines_skeleton[l,0,:],lines_skeleton[l,1,:],lines_skeleton[l,2,:], c='red')
        # Limits used for GT plots
        ax.set_xlim3d([-600, 600])
        ax.set_ylim3d([-600, 600])
        ax.set_zlim3d([0, 1400])
        plt.show()
        # Put visualization, debug prints, etc... here
        cv2.imshow("events", self.image)
        cv2.waitKey(10)
        return True
    # def load_model(self):
    #     self.model = utilities.load_model(checkpoint_path, "MargiposeEstimator", core=dhpcore).eval().double()
    #     factory = MinimalConstructor()#Joints3DConstructor()
    #     factory.set_dataset_core(dhpcore)
    #     tester= factory.get_dataset()



if __name__ == '__main__':

    # prepare and configure the resource finder
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("event-driven")
    rf.setDefaultConfigFile("exampleModule.ini")
    rf.configure(sys.argv)

    # create the module
    module = ExampleModule()
    module.runModule(rf)
