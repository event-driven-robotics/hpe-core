
#include <Python.h>
#include <hpe-core/e2vid.h>
#include <yarp/os/all.h>
#include <yarp/sig/Image.h>

using namespace yarp::os;
using namespace yarp::sig;


class E2VidExampleModule : public RFModule, public Thread {

private:

    vReadPort<vector<AE>> input_port;
    BufferedPort<ImageOf<PixelRgb>> output_port;

    deque<AE> out_queue;

    int sensor_height;
    int sensor_width;
    int window_size;
    float events_per_pixel;

    hpecore::E2Vid e2vid;

public:

    E2VidExampleModule() {}

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

        e2vid.init(sensor_height, sensor_width, window_size, events_per_pixel);

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
        e2vid.close();
        return true;
    }

    void onStop()
    {
        // close ports
        input_port.close();
        output_port.close();

        // kill python's interpreter
        e2vid.close();
    }

    // synchronous thread
    virtual bool updateModule() {

        // TODO: check number of events in out_queue?

        // run e2vid on deque of events
        // ...
        // out_queue.clear();


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

            const vector<AE> * q = input_port.read(yarpstamp);
            if(!q || Thread::isStopping()) return;

            for(auto &qi : *q) {

                // TODO: add check on number of events?
                // if(out_queue.size() > max_events_num)
                //     continue;

                out_queue.push_back(qi);
            }
        }
    }
};


int main(int argc, char *argv[]) {

    Py_Initialize();

    printf("python home: %ls\n", Py_GetPythonHome());
    printf("program name: %ls\n", Py_GetProgramName());
    printf("get path: %ls\n", Py_GetPath());
    printf("get prefix: %ls\n", Py_GetPrefix());
    printf("get exec prefix: %ls\n", Py_GetExecPrefix());
    printf("get prog full path: %ls\n", Py_GetProgramFullPath());

    PyRun_SimpleString("import sys");
    printf("path: ");
    PyRun_SimpleString("print(sys.path)");
    printf("version: ");
    PyRun_SimpleString("print(sys.version)");

    printf("E2VID_PYTHON_DIR: %s\n", E2VID_PYTHON_DIR);

    hpecore::E2Vid e2vid;
    e2vid.init(260, 346, 7500, 0.35);

    return 0;
}
