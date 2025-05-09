#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/cv/Cv.h>
#include <event-driven/core.h>
#include <event-driven/algs.h>

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;

class eroser : public RFModule {

private:

    ev::EROS eros;
    ev::window<ev::AE> input_port;
    yarp::os::BufferedPort<ImageOf<PixelMono> > output_port;

    resolution res;
    Stamp yarpstamp;
    int freq;

    std::thread erosloop;

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {
        //set the module name used to name ports
        setName((rf.check("name", Value("/eroser")).asString()).c_str());

        /* initialize yarp network */
        yarp::os::Network yarp;
        if(!yarp.checkNetwork(2.0)) {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }

        //open io ports
        if(!input_port.open(getName("/AE:i"))) {
            yError() << "Could not open input port";
            return false;
        }

        if(!output_port.open(getName("/img:o"))) {
            yError() << "Could not open output port";
            return false;
        }

        //read flags and parameters
        res.height = rf.check("height", Value(480)).asInt32();
        res.width = rf.check("width", Value(640)).asInt32();

        eros.init(res.width, res.height, 7, 0.3);

        freq = rf.check("f", Value(50)).asInt32();

        erosloop = std::thread([this]{events2eros();});

        //start the asynchronous and synchronous threads
        return true;
    }

    virtual double getPeriod() override
    {
        return 1.0/freq; //period of synchrnous thread
    }

    bool interruptModule() override
    {
        //if the module is asked to stop ask the asynchrnous thread to stop
        input_port.stop();
        erosloop.join();
        output_port.close();
        return true;
    }

    //synchronous thread
    virtual bool updateModule()
    {

        //static cv::Mat cv_image(res.height, res.width, CV_8U);
        static cv::Mat eros8U;

        eros.getSurface().convertTo(eros8U, CV_8U);

        cv::GaussianBlur(eros8U, eros8U, cv::Size(5, 5), 0, 0);

        output_port.setEnvelope(yarpstamp);
        output_port.prepare().copy(yarp::cv::fromCvMat<PixelMono>(eros8U));
        output_port.write();
        return true;
    }

    //asynchronous thread run forever
    void events2eros() {

        while (input_port.isRunning()) {
            ev::info packet_info = input_port.readAll(true);
            for(auto &v : input_port)
                eros.update(v.x, v.y);
        }
    }

};

int main(int argc, char * argv[])
{
    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.configure( argc, argv );

    /* create the module */
    eroser instance;
    return instance.runModule(rf);
}