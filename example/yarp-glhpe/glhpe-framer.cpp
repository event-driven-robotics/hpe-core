#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/cv/Cv.h>
#include <event-driven/all.h>
#include <mutex>
#include <hpe-core/volumes.h>

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;


class glhpewindower : public RFModule, public Thread {

private:
    deque<AE> window_queue;
    vReadPort<vector<AE> > input_port;
    BufferedPort<ImageOf<PixelMono> > output_port;
    std::mutex m;

    int window_size;
    resolution res;

public:

    glhpewindower() {}

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        //set the module name used to name ports
        setName((rf.check("name", Value("/glhpewindow")).asString()).c_str());

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
        window_size = rf.check("window", Value(7500)).asInt();
        res.height = rf.check("height", Value(300)).asInt();
        res.width = rf.check("width", Value(400)).asInt();

        //start the asynchronous and synchronous threads
        return Thread::start();
    }

    virtual double getPeriod()
    {
        return 0.1; //period of synchrnous thread
    }

    bool interruptModule()
    {
        //if the module is asked to stop ask the asynchrnous thread to stop
        return Thread::stop();
    }

    void onStop()
    {
        //when the asynchrnous thread is asked to stop, close ports and do
        //other clean up
        input_port.close();
        output_port.close();
    }

    //synchronous thread
    virtual bool updateModule()
    {
        //add any synchronous operations here, visualisation, debug out prints
        //ImageOf<PixelMono> &image = output_port.prepare();
        //image.resize(res.width, res.height);
        //image.zero();

        cv::Mat cv_image(res.height, res.width, CV_8U);

        m.lock();
        hpecore::createCountImage(window_queue, cv_image);
        m.unlock();
        hpecore::varianceNormalisation(cv_image);

        output_port.prepare().copy(yarp::cv::fromCvMat<PixelMono>(cv_image));
        output_port.write();
        return Thread::isRunning();
    }

    //asynchronous thread run forever
    void run() {
        Stamp yarpstamp;

        while (true) {
            const vector<AE>* q = input_port.read(yarpstamp);
            if(!q || Thread::isStopping()) return;

            m.lock();
            for (auto& qi : *q)
                window_queue.push_back(qi);

            while(window_queue.size() > window_size)
                window_queue.pop_front();
            m.unlock();
        }
    }
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
    rf.configure( argc, argv );

    /* create the module */
    glhpewindower instance;
    return instance.runModule(rf);
}
