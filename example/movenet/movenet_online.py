from lib import init, Data, MoveNet, Task

from config import cfg
from lib.utils.utils import arg_parser, image_show

import sys
import yarp
import numpy as np
import cv2
import os

# Copied from Predict.py


def main(cfg):
    init(cfg)

    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')

    data = Data(cfg)
    test_loader = data.getTestDataloader()

    run_task = Task(cfg, model)
    run_task.modelLoad("models/h36m_finetuned.pth")

    run_task.predict(test_loader, cfg['predict_output_path'])



# Copied from GL-HPE
class MovenetModule(yarp.RFModule):

    def __init__(self, cfg):
        yarp.RFModule.__init__(self)
        self.input_port = yarp.BufferedPortImageMono()
        self.output_port = yarp.BufferedPortImageMono()
        self.stamp = yarp.Stamp()
        self.counter = 0
        self.image_w = 640 # Size of image expected from the framer.
        self.image_h = 480 #
        # self.np_input = None
        self.yarp_image = yarp.ImageMono()
        self.yarp_image_out = yarp.ImageRgb()

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
        # self.model = None
        # self.read_image = None

    def configure(self, rf):

        # Initialise YARP
        yarp.Network.init()
        if not yarp.Network.checkNetwork(2):
            print("Could not find network! Run yarpserver and try again.")
            exit(-1)

        # set the module name used to name ports
        self.setName((rf.check("name", yarp.Value("/MovenetModule")).asString()))

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

        # if len(input_image.shape) == 2:
        #     input_image = np.expand_dims(input_image, -1)
        #     input_image = np.expand_dims(input_image, 0)

        # Predict the pose
        # torch_image = torch.from_numpy(input_image)
        pre = self.run_task.predict_online(input_image, cfg['predict_output_path'])


        # Visualize the result
        k = image_show(input_image,pre)

        # cv2.imshow("output", img)
        # print(img.shape)
        # k = cv2.waitKey(10)
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
    cfg = arg_parser(cfg)

    # create the module
    module = MovenetModule(cfg)
    module.runModule(rf)
