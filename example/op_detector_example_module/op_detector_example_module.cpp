
#include <opencv2/opencv.hpp>
#include <yarp/cv/Cv.h>
#include <yarp/os/all.h>
#include <yarp/sig/Image.h>
#include <hpe-core/openpose_detector.h>
#include <hpe-core/utility.h>

using namespace yarp::os;
using namespace yarp::sig;

class OPDetectorExampleModule : public RFModule {

private:

    BufferedPort<ImageOf<PixelMono>> input_port;
    BufferedPort<ImageOf<PixelRgb>> output_port_frame;
    BufferedPort<Bottle> output_port_skeleton;

    std::string models_path;
    std::string pose_model;

    hpecore::OpenPoseDetector detector;

public:

    OPDetectorExampleModule() {}

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        if (!yarp::os::Network::checkNetwork(2.0)) {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }
        //set the module name used to name ports
        setName((rf.check("name", Value("/op_detector_example_module")).asString()).c_str());

        //open io ports
        if(!input_port.open(getName() + "/img:i")) {
            yError() << "Could not open input port";
            return false;
        }

        if(!output_port_frame.open(getName() + "/img:o")) {
            yError() << "Could not open output frame port";
            return false;
        }

        if(!output_port_skeleton.open(getName() + "/skl:o")) {
            yError() << "Could not open output skeleton port";
            return false;
        }

        //read flags and parameters
        std::string default_models_path = "/openpose/models";
        models_path = rf.check("models_path", Value(default_models_path)).asString();

        std::string default_pose_model = "BODY_25";
        pose_model = rf.check("pose_model", Value(default_pose_model)).asString();

        detector.init(models_path, pose_model);

        return true;
    }

    virtual double getPeriod()
    {
        //run the module as fast as possible. Only as fast as new images are
        //available and then limited by how fast OpenPose takes to run
        return 0; 
    }

    bool interruptModule()
    {
        //if the module is asked to stop ask the asynchronous thread to stop
        input_port.interrupt();
        output_port_frame.interrupt();
        output_port_skeleton.interrupt();
        return true;
    }

    bool close()
    {
        //when the asynchronous thread is asked to stop, close ports and do other clean up
        detector.stop();
        input_port.close();
        output_port_frame.close();
        output_port_skeleton.close();
        return true;
    }

    //synchronous thread
    virtual bool updateModule()
    {
        // read greyscale image from yarp
        ImageOf<PixelMono>* img_yarp = input_port.read();

        Stamp timestamp;
        input_port.getEnvelope(timestamp);

        if (img_yarp == nullptr)
            return false;

        double tic = Time::now();
        static double t_accum = 0.0;
        static int t_count = 0;

        //convert image to OpenCV RGB format
        cv::Mat img_cv = yarp::cv::toCvMat(*img_yarp);
        cv::Mat rgbimage;
        cv::cvtColor(img_cv, rgbimage, cv::COLOR_GRAY2BGR);

        //call the openpose detector
        hpecore::skeleton pose = detector.detect(rgbimage);

        //calculate the running frequency
        t_accum += Time::now() - tic;
        t_count++;
        if(t_accum > 2.0) {
            yInfo() << "Running happily at an upper bound of" << (int)(t_count / t_accum) << "Hz";
            t_accum = 0.0;
            t_count = 0;
        }

        // write opnepose's output frame to the output port
        output_port_frame.prepare().copy(yarp::cv::fromCvMat<PixelBgr>(rgbimage));
//        output_port_frame.setEnvelope(timestamp);
        output_port_frame.write();

//        // write opnepose's output skeleton to the output port
//        output_port_skeleton.prepare().copy(yarp::cv::fromCvMat<PixelBgr>(rgbimage));
//        output_port_skeleton.setEnvelope(timestamp);
//        output_port_skeleton.write();

//        //visualisation
//        cv::imshow("OpenPose", rgbimage);
//        cv::waitKey(1);

        return true;
    }
};

int main(int argc, char * argv[])
{
    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setVerbose( false );
    rf.configure( argc, argv );

    /* create the module */
    OPDetectorExampleModule instance;
    return instance.runModule(rf);
}
