import sys
sys.path.append('/home/aglover/install/lib/python3/dist-packages')
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

class movenetModule(yarp.RFModule):

    def __init__(self):
        yarp.RFModule.__init__(self)
        self.output_port = yarp.BufferedPortBottle()
        self.stamp = yarp.Stamp()


    def configure(self, rf):

        if not self.output_port.open(self.getName() + "/SKLT:o"):
            print("Could not open output port")
            return False

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

        #read data
        input_port.getEnvelope(stamp)

        #invoke movenet

        #convert skeleton to bottle format
        yarp.Bottle message = output_port.prepare()
        message.clear()
        message.addString('SKLT')
        yarp.Bottle joints = message.addList()
        for j in joint_list
            joints.addDouble(j.u)
            joints.addDouble(j.v)
        output_port.setEnvelop(stamp)
        output_port.write()
        
        return True


if __name__ == '__main__':
    # prepare and configure the resource finder
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("event-driven")
    # rf.setDefaultConfigFile("exampleModule.ini")
    rf.configure(sys.argv)

    # create the module
    module = movenetModule()
    module.runModule(rf)
