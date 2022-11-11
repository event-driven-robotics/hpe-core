#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/cv/Cv.h>
#include <event-driven/all.h>
#include <mutex>
#include <hpe-core/representations.h>
#include <hpe-core/motion_estimation.h>

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;


class eroser : public RFModule, public Thread {

private:

    hpecore::EROS eros;
    vReadPort<vector<AE> > input_port;
    BufferedPort<ImageOf<PixelMono> > output_port;
    std::mutex m;
    hpecore::pwvelocity pw_velocity;

    resolution res;
    Stamp yarpstamp;
    int freq;

public:

    eroser() {}

    virtual bool configure(yarp::os::ResourceFinder& rf)
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

        Network::connect("/atis3/AE:o", getName("/AE:i"), "fast_tcp");

        //read flags and parameters
        res.height = rf.check("height", Value(480)).asInt32();
        res.width = rf.check("width", Value(640)).asInt32();

        // eros.init(res.width, res.height, 7, 0.3);

        cv::Size image_size = cv::Size(res.width, res.height);
        pw_velocity.setParameters(image_size, 7, 0.3, 0.01);

        freq = rf.check("f", Value(50)).asInt32();

        //start the asynchronous and synchronous threads
        return Thread::start();
    }

    virtual double getPeriod()
    {
        return 1.0/freq; //period of synchrnous thread
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

    string type2str(int type)
    {
        string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch (depth)
        {
        case CV_8U:
            r = "8U";
            break;
        case CV_8S:
            r = "8S";
            break;
        case CV_16U:
            r = "16U";
            break;
        case CV_16S:
            r = "16S";
            break;
        case CV_32S:
            r = "32S";
            break;
        case CV_32F:
            r = "32F";
            break;
        case CV_64F:
            r = "64F";
            break;
        default:
            r = "User";
            break;
        }

        r += "C";
        r += (chans + '0');

        return r;
    }
    //synchronous thread
    virtual bool updateModule()
    {

        static cv::Mat cv_image(res.height, res.width, CV_8U);
        static cv::Mat eros_copy;

        m.lock();
        // eros.getSurface().copyTo(eros_copy);
        auto aux = pw_velocity.queryEROS().clone();
        aux.convertTo(eros_copy, CV_8U,255,0);
        m.unlock();

        // string ty =  type2str( eros_copy.type() );
        // printf("Matrix: %s %dx%d \n", ty.c_str(), eros_copy.cols, eros_copy.rows );

        cv::GaussianBlur(eros_copy, cv_image, cv::Size(5, 5), 0, 0);

        output_port.setEnvelope(yarpstamp);
        output_port.prepare().copy(yarp::cv::fromCvMat<PixelMono>(cv_image));
        output_port.write();

        // cv::imshow("EROS-framer", cv_image);
        // // cv::imshow("AUX", aux);
        // cv::waitKey(1);

        return Thread::isRunning();
    }

    //asynchronous thread run forever
    void run() {

        const vector<AE> *q;
        Stamp ystamp;
        double t0 = Time::now(), // initial time
            tnow = t0;

        while (!Thread::isStopping()) {
            // const vector<AE>* q = input_port.read(yarpstamp);
            // if(!q || Thread::isStopping()) return;

            // m.lock();
            // for (auto& qi : *q)
            //     eros.update(qi.x, qi.y);
            // m.unlock();
            tnow = Time::now() - t0;
            int nqs = input_port.queryunprocessed();
            m.lock();
            for (auto i = 0; i < nqs; i++)
            {
                auto q = input_port.read(ystamp);
                if (!q)
                    return;
                
                pw_velocity.update<vector<AE>>(*q, tnow);
            }
            m.unlock();
        }
    }

};

int main(int argc, char * argv[])
{
    /* initialize yarp network */
    yarp::os::Network yarp;
    if(!yarp.checkNetwork(2.0)) {
        std::cout << "Could not connect to YARP" << std::endl;
        return false;
    }

    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.configure( argc, argv );

    /* create the module */
    eroser instance;
    return instance.runModule(rf);
}
