from lib import init, MoveNet, Task

from config import cfg
from lib.utils.utils import arg_parser
from lib.task.task_tools import image_show


import sys
import yarp
import numpy as np
import cv2
import torch


dev = True

if dev:
    import glob
    import os


class MovenetModule(yarp.RFModule):

    def __init__(self, cfg):
        yarp.RFModule.__init__(self)
        self.input_port = yarp.BufferedPortImageMono()
        self.output_port = yarp.BufferedPortImageMono()
        self.stamp = yarp.Stamp()
        # self.counter = 0
        self.image_w = 640 # Size of image expected from the framer.
        self.image_h = 480 #
        # self.np_input = None
        self.yarp_image = yarp.ImageMono()
        self.yarp_image_out = yarp.ImageRgb()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = "models/h36m_finetuned.pth"

        self.resultsPath =  '/outputs'
        self.image_w_model = 192 # Size of the image expected by the model
        self.image_h_model = 192 #
        self.output_w = 640  # Size of the image expected by yarp
        self.output_h = 480  #
        self.fname = None
        self.fname_ts = None
        self.last_timestamp = 0.0
        self.cfg = cfg
        if dev:
            self.tester_path = '/media/Data/data/eros_samples_live_220408/' #path to a folder with images.


    def configure(self, rf):

        # Initialise YARP
        yarp.Network.init()
        if not yarp.Network.checkNetwork(2):
            print("Could not find network! Run yarpserver and try again.")
            exit(-1)

        # set the module name used to name ports
        self.setName((rf.check("name", yarp.Value("/movenet")).asString()))

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

        self.model = MoveNet(num_classes=self.cfg["num_classes"],
                        width_mult=self.cfg["width_mult"],
                        mode='train')
        self.run_task = Task(cfg, self.model)
        self.run_task.modelLoad("models/h36m_finetuned.pth")

        if dev:
            self.files = glob.glob(self.tester_path+"/*.jpg")
            self.file_counter = 0

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


        if dev:
            np_input = cv2.imread(self.files[self.file_counter])
            print(np_input.shape)
            np_input = cv2.cvtColor(np_input, cv2.COLOR_BGR2GRAY)
            print(np_input.shape)
        else:

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

            # End of input file
            # if self.last_timestamp == stamp:
            #     return False

            # self.counter += 1  # can be used to interrupt the program
            self.yarp_image.copy(read_image)

        input_image = np.copy(np_input)

        input_image_resized = np.zeros([1, 3, self.image_h_model,self.image_w_model])
        # print(input_image_resized.shape)

        input_image = cv2.resize(input_image,(self.image_h_model,self.image_w_model))
        input_image_resized[0, 0,:,:] = input_image[:,:]
        input_image_resized[0, 1,:,:] = input_image[:,:]
        input_image_resized[0, 2,:,:] = input_image[:,:]

        input_image_resized = input_image_resized.astype(np.float32)

        # Predict the pose

        pre = self.run_task.predict_online(input_image_resized)

        # Visualize the result
        image_show(input_image,pre)
        k = cv2.waitKey(1)
        if dev:
            self.file_counter += 1
            if self.file_counter >= 19:
                return False
        else:
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
    cfg = arg_parser(cfg)

    # create the module
    module = MovenetModule(cfg)
    module.runModule(rf)
