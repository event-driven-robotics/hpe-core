
import yarp
import sys
import numpy as np
import threading
import cv2

import e2vid_config
from e2vid import E2Vid


class E2VidExampleModule(yarp.RFModule):

    def __init__(self):
        yarp.RFModule.__init__(self)
        self.image = np.zeros((e2vid_config.sensor_height, e2vid_config.sensor_width), dtype=np.uint8)
        self.image_buf = np.zeros((e2vid_config.sensor_height, e2vid_config.sensor_width), dtype=np.uint8)
        self.events = np.zeros((7500, 4), dtype=np.float64)
        self.events_buf = np.zeros((7500, 4), dtype=np.float64)
        self.input_port = yarp.BufferedPortBottle()

        self.out_buf_image = yarp.ImageMono()
        self.out_buf_image.resize(e2vid_config.sensor_width, e2vid_config.sensor_height)

        self.out_buf_image.setExternal(self.image.data, self.image.shape[1], self.image.shape[0])

        # self.output_port = yarp.BufferedPortImageMono()
        self.output_port = yarp.Port()
        self.rpc_port = yarp.RpcServer()
        cv2.namedWindow("events", cv2.WINDOW_NORMAL)
        self.mutex = threading.Lock()

        self.e2vid = E2Vid(e2vid_config)
        # self.e2vid.model_share_memory()

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

        print('***************************')
        print(self.image.shape)

        self.output_port.write(self.out_buf_image)

        cv2.imshow("events", self.image)
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

            self.events_buf = np.concatenate((timestamps, x, y, pol), axis=1)
            self.image_buf = self.e2vid.predict_grayscale_frame(self.events_buf)
            self.mutex.acquire()
            self.image = self.image_buf.copy()  # self.image is a shared resource between threads
            # self.events = self.events_buf.copy()
            self.mutex.release()


if __name__ == '__main__':
    # Initialise YARP
    yarp.Network.init()
    if not yarp.Network.checkNetwork(2):
        print("Could not find network! Run yarpserver and try again.")
        exit(-1)

    # prepare and configure the resource finder
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("eventdriven")
    # rf.setDefaultConfigFile("e2vidExampleModule.ini")
    rf.configure(sys.argv)

    # create the module
    module = E2VidExampleModule()
    module.runModule(rf)
