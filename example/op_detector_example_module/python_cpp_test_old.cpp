
#include <Python.h>
//#include <hpe-core/e2vid.h>
#include "e2vid.h"
//#include <hpe-core/utility.h>
#include <yarp/os/all.h>
#include <yarp/sig/Image.h>
#include <event-driven/all.h>
#include <vector>

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;


class E2VidExampleModule: public RFModule/*, public Thread*/ {

private:

    vReadPort<std::vector<AE>> input_port;
    BufferedPort<ImageOf<PixelRgb>> output_port;

    // contrary to other module examples where out_queue is of class std::deque,
    // here out_queue must be converted to an object representing a numpy array,
    // and for this purpose elements must be stored contiguously in memory (possible
    // only with std::vector)
    std::vector<AE> out_queue;

    int sensor_height;
    int sensor_width;
    int window_size;
    float events_per_pixel;

    E2Vid e2vid;

public:

    E2VidExampleModule()
    {
//        setName("/e2vid_online_module");
//
//        //open io ports
//        if(!input_port.open(getName() + "/AE:i")) {
//            yError() << "Could not open input port";
//        }
//
//        if(!output_port.open(getName() + "/img:o")) {
//            yError() << "Could not open input port";
//        }
//
//        sensor_height = 260;
//        sensor_width = 346;
//        window_size = 7500;
//        events_per_pixel = 0.35;
//
//        e2vid.init(sensor_height, sensor_width, window_size, events_per_pixel);
    }

    void init_e2vid()
    {
        e2vid.init(260, 346, 7500, 0.35);
    }

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        //set the module name used to name ports
        setName((rf.check("name", Value("/e2vid_online_module")).asString()).c_str());

        //open io ports
        if(!input_port.open(getName() + "/AE:i")) {
            yError() << "Could not open input port";
            return false;
        }

        if(!output_port.open(getName() + "/img:o")) {
            yError() << "Could not open input port";
            return false;
        }

        // read flags and parameters
        int default_sensor_height = 260;
        sensor_height = rf.check("sensor_height", Value(default_sensor_height)).asInt();

        int default_sensor_width = 346;
        sensor_width = rf.check("sensor_width", Value(default_sensor_width)).asInt();

        int default_window_size = 7500;
        window_size = rf.check("window_size", Value(default_window_size)).asInt();

        float default_events_per_pixel = 0.35;
        events_per_pixel = rf.check("events_per_pixel", Value(default_events_per_pixel)).asFloat64();

//        yInfo() << "************************ before e2vid.init";
//        e2vid.init(sensor_height, sensor_width, window_size, events_per_pixel);
//        yInfo() << "************************ after e2vid.init";

        return true;
    }

    virtual double getPeriod()
    {
        // run the module as fast as possible. Only as fast as new images are
        // available and then limited by how fast e2vid takes to run
        return 0;
    }

    bool interruptModule()
    {
        // close ports
        input_port.interrupt();
        output_port.interrupt();

        // kill python's interpreter
//        e2vid.close();
        return true;
    }

    void onStop()
    {
        // close ports
        input_port.close();
        output_port.close();

        // kill python's interpreter
//        e2vid.close();
    }

    void set_e2vid(E2Vid& instance)
    {
        e2vid = instance;
    }

    // synchronous thread
    virtual bool updateModule()
    {
        yInfo() << "---------------------------------- updateModule()";

        // TODO: check number of events in out_queue?

        // run e2vid on deque of events
        cv::Mat grayscale_frame;
//        e2vid.predict_grayscale_frame(out_queue, grayscale_frame);
        out_queue.clear();


//        //read greyscale image from yarp
//        ImageOf<PixelMono>* img_yarp = input_port.read();
//        if (img_yarp == nullptr)
//            return false;
//
//        double tic = Time::now();
//        static double t_accum = 0.0;
//        static int t_count = 0;
//
//        //convert image to OpenCV RGB format
//        cv::Mat img_cv = yarp::cv::toCvMat(*img_yarp);
//        cv::Mat rgbimage;
//        cv::cvtColor(img_cv, rgbimage, cv::COLOR_GRAY2BGR);
//
//        //call the openpose detector
//        hpecore::skeleton pose = detector.detect(rgbimage);
//
//        //calculate the running frequency
//        t_accum += Time::now() - tic;
//        t_count++;
//        if(t_accum > 2.0) {
//            yInfo() << "Running happily at an upper bound of" << (int)(t_count / t_accum) << "Hz";
//            t_accum = 0.0;
//            t_count = 0;
//        }
//
//        //visualisation
//        cv::imshow("OpenPose", rgbimage);
//        cv::waitKey(1);

        return true;
    }

    // asynchronous thread runs forever
    void run()
    {
        Stamp yarpstamp;

        while(true) {

            const std::vector<AE> * q = input_port.read(yarpstamp);
            if(!q/* || Thread::isStopping()*/) return;

            yInfo() << "******************** run() - events received - out_queue.size(): " << out_queue.size();

            for(auto &qi : *q) {

                // TODO: add check on number of events?
                // if(out_queue.size() > max_events_num)
                //     continue;

                out_queue.push_back(qi);
            }
        }
    }
};


int main(int argc, char * argv[])
{
    // TODO: DOESN'T WORK?!
    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
//    rf.setVerbose( false );
//    rf.configure( argc, argv );

    /* create the module */
//    E2VidExampleModule instance;

//    yInfo() << "-------------------------------------- initializing e2vid";
//    instance.init_e2vid();
//    yInfo() << "--------------------------------------finished";

    E2Vid e2vid;
    e2vid.init(260, 346, 7500, 0.35);
//    instance.set_e2vid(e2vid);
    return 0;
//    return instance.runModule(rf);

//    // TODO: WORKS?!
//    hpecore::E2Vid e2vid;
//    e2vid.init(260, 346, 7500, 0.35);
}
