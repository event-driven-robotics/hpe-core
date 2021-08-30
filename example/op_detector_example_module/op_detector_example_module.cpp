
#include <opencv2/opencv.hpp>
#include <yarp/cv/Cv.h>
#include <yarp/os/all.h>
#include <yarp/sig/Image.h>
#include <event-driven/all.h>

#include <hpe-core/openpose_detector.h>
#include <hpe-core/utility.h>

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;

class OPDetectorExampleModule : public RFModule {

private:

    BufferedPort<ImageOf<PixelRgb>> input_port;
    BufferedPort<ImageOf<PixelRgb>> output_port;

    std::string models_path;
    std::string pose_model;

    hpecore::OpenPoseDetector detector;

public:

    OPDetectorExampleModule() {}

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        //set the module name used to name ports
        setName((rf.check("name", Value("/op_detector_example_module")).asString()).c_str());

        //open io ports
        if(!input_port.open(getName() + "/AE:i")) {
            yError() << "Could not open input port";
            return false;
        }

        if(!output_port.open(getName() + "/AE:o")) {
            yError() << "Could not open input port";
            return false;
        }

        //read flags and parameters
        std::string default_models_path = "";
        models_path = rf.check("models_path", Value(default_models_path)).asString();

        std::string default_pose_model = "BODY_25";  // TODO: get the default string from openpose header files
        pose_model = rf.check("pose_model", Value(default_pose_model)).asString();

        detector.init(models_path, pose_model);
//        if(detector.init(models_path, pose_model))
//            //start the asynchronous and synchronous threads
//            return Thread::start();
//        else
//            Thread::stop();
    }

    virtual double getPeriod()
    {
        return 1.0; //period of synchronous thread
    }

    bool interruptModule()
    {
        //if the module is asked to stop ask the asynchronous thread to stop
//        return Thread::stop();
        yInfo() << "Interrupting module ...";
        return RFModule::interruptModule();
    }

    bool close()
    {
        //when the asynchronous thread is asked to stop, close ports and do other clean up
        yInfo() << "Closing the module ...";
        input_port.close();
        output_port.close();
        yInfo() << "Module closed!";
        return RFModule::close();
    }

    //synchronous thread
    virtual bool updateModule()
    {
//        ImageOf<PixelRgb>* img_yarp = input_port.read();
//        if (img_yarp == nullptr)
//        {
//            yError() << "Failed to read yarp image";
//        }
//
//        cv::Mat img_cv = toCvMat(*img_yarp);
//        yInfo() << "yarp image converted to cv::Mat";

        // read image from directory
        cv::Mat img_cv = cv::imread("/workspace/data/dhp19/events_preprocessed/S1_1_1/0/reconstruction/frame_0000003065.png");
        hpecore::skeleton pose = detector.detect(img_cv);
        yInfo() << "pose detector has finished";
//        std::cout << "pose detected" << std::endl;
//        std::cout << pose << std::endl;

        // TODO: plot pose onto image

        output_port.prepare().copy(yarp::cv::fromCvMat<PixelBgr>(img_cv));
        output_port.write();

//        return Thread::isRunning();
        return true;
    }

//    void run()
//    {
//
//        while(true) {
//        }
//    }
};

int main(int argc, char * argv[])
{
    /* initialize yarp network */
    yarp::os::Network yarp;
    if(!yarp.checkNetwork(2)) {
        std::cout << "Could not connect to YARP" << std::endl;
        return false;
    }

    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setVerbose( false );
    rf.setDefaultContext( "eventdriven" );
    rf.setDefaultConfigFile( "op_detector_example_module.ini" );
    rf.configure( argc, argv );

    /* create the module */
    OPDetectorExampleModule instance;
    return instance.runModule(rf);
}
