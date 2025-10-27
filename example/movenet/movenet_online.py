from pycore.moveenet import MoveNet, Task

from config import cfg
from pycore.moveenet.utils.utils import arg_parser
from pycore.moveenet.task.task_tools import write_output

import sys
import yarp
import numpy as np

import datetime

class MovenetModule(yarp.RFModule):

    def __init__(self, cfg):
        yarp.RFModule.__init__(self)
        self.input_port = yarp.BufferedPortImageMono()
        self.output_port = yarp.Port()
        self.stamp = yarp.Stamp()

        # Size of image expected from the framer.
        self.image_w = cfg['w']
        self.image_h = cfg['h']

        # Size of the image expected by the model
        self.image_w_model = 192
        self.image_h_model = 192

        self.yarp_image = yarp.ImageMono()
        self.yarp_sklt_out = yarp.Bottle()
        self.checkpoint_path = cfg['checkpoint_path']
        self.resultsPath = '/outputs'

        self.fname = None
        self.fname_ts = None
        self.last_timestamp = 0.0
        self.cfg = cfg

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

        self.model = MoveNet(num_classes=self.cfg["num_classes"],
                             width_mult=self.cfg["width_mult"],
                             mode='train')
        self.run_task = Task(cfg, self.model)
        self.run_task.modelLoad(self.checkpoint_path)

        # open io ports
        if not self.input_port.open(self.getName() + "/img:i"):
            print("Could not open input port")
            return False
        if not self.output_port.open(self.getName() + "/sklt:o"):
            print("Could not open output port")
            return False

        return True

    def getPeriod(self):
        # period of synchronous thread, return 0 update module called as fast as it can
        return 0

    def interruptModule(self):
        # interrupting all the ports
        self.input_port.interrupt()
        self.output_port.interrupt()
        return True

    def close(self):
        # closing ports
        self.input_port.close()
        self.output_port.close()
        # cv2.destroyAllWindows()
        return True

    # synchronous update called every get period seconds.
    def updateModule(self):

        # Preparing input and output image buffers
        np_input = np.ones((self.image_h, self.image_w), dtype=np.uint8)
        self.yarp_image.resize(self.image_w, self.image_h)
        self.yarp_image.setExternal(np_input.data, np_input.shape[1], np_input.shape[0])

        # Read the image
        read_image = self.input_port.read()
        if read_image is None:
            return False
        self.input_port.getEnvelope(self.stamp)

        stamp_in = self.stamp.getCount() + self.stamp.getTime()

        self.yarp_image.copy(read_image)

        input_image = np.copy(np_input)
        t0 = datetime.datetime.now()

        # Predict the pose
        pre = self.run_task.predict_online(input_image)

        # latency
        t1 = datetime.datetime.now()
        delta = t1-t0
        latency = delta.microseconds / 1000

        self.last_timestamp = stamp_in
        # print(stamp_in)
        if self.cfg['write_output']:
            write_output('file.csv', pre['joints'], timestamp=stamp_in)

        # Export output skeleton
        stamp = yarp.Stamp(0, latency)

        self.output_port.setEnvelope(stamp)
        self.yarp_sklt_out.clear()

        out_sklt = np.concatenate((pre['joints'], pre['confidence']))
        self.yarp_sklt_out.addString('SKLT')
        temp_list = self.yarp_sklt_out.addList()
        for i in out_sklt:
            temp_list.addFloat64(i)

        self.output_port.write(self.yarp_sklt_out)

        return True


if __name__ == '__main__':
    # prepare and configure the resource finder
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("event-driven")

    rf.configure(sys.argv)
    cfg = arg_parser(cfg)

    # create the module
    module = MovenetModule(cfg)
    module.runModule(rf)
