#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <event-driven/all.h>
#include <mutex>

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
        res.height = rf.check("height", Value(260)).asInt();
        res.width = rf.check("width", Value(346)).asInt();

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
        ImageOf<PixelMono> &image = output_port.prepare();
        image.resize(res.width, res.height);
        image.zero();

        m.lock();

        int count_unique = 0;
        for(auto &v : window_queue) 
        {
            PixelMono &p = image(v.x, v.y);
            if(p == 0) count_unique++;
            p += 1;
        }

        m.unlock();

        //count unique pixels
        double mean_pval = (double)window_queue.size() / count_unique;

        double var = 0;
        for(auto x = 0; x < res.width; x++) {
            for(auto y = 0; y < res.height; y++) {
                PixelMono &p = image(x, y);
                if(p > 0) {
                    double e = p - mean_pval;
                    var += e*e;
                }
            }
        }
        var /= count_unique;
        double sigma = sqrt(var);
        constexpr double threshold = 0.1/255;
        if(sigma < threshold) sigma = threshold;
        double scale_factor = 255.0 / (3.0 * sigma);

        for (auto x = 0; x < res.width; x++) {
            for(auto y = 0; y < res.height; y++) {
                PixelMono &p = image(x, y);
                if(p > 0) {
                    double v = p * scale_factor;
                    if(v > 255.0) v = 255.0;
                    if(v < 0.0) v = 0.0;
                    p = (unsigned char)v;
                }
            }
        }

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
