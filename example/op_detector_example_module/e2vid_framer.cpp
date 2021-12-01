#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/cv/Cv.h>
#include <event-driven/all.h>
#include <mutex>

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;


class e2vidPacker : public RFModule, public Thread {

private:
    deque<AE> window_queue;
    vReadPort<vector<AE> > input_port;
    vWritePort output_port;
    Stamp yarpstamp;
    std::mutex m;

    bool send_update{false};
    int window_size;

public:

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        //set the module name used to name ports
        setName((rf.check("name", Value("/e2vidPacker")).asString()).c_str());

        /* initialize yarp network */
        if (!yarp::os::Network::checkNetwork(2)) {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }

        //open io ports
        if(!input_port.open(getName("/AE:i"))) {
            yError() << "Could not open input port";
            return false;
        }

        if(!output_port.open(getName("/AE:o"))) {
            yError() << "Could not open output port";
            return false;
        }

        //read flags and parameters
        window_size = rf.check("window", Value(7500)).asInt();

        //start the asynchronous and synchronous threads
        return Thread::start();
    }

    virtual double getPeriod()
    {
        return 0.2; //period of synchrnous thread
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

        m.lock();
        deque<AE> fakeq = window_queue;
	m.unlock();
        if(send_update && fakeq.size())
            	output_port.write(fakeq, yarpstamp);
        send_update = false;

        return Thread::isRunning();
    }

    //asynchronous thread run forever
    void run() {
        
        while (true) {
            //if(input_port.queryunprocessed() > 10)
            //      yInfo() <<  "delay" << input_port.queryunprocessed() << "packets";
            const vector<AE>* q = input_port.read(yarpstamp);
            if(!q || Thread::isStopping()) return;

            m.lock();
            for (auto& qi : *q)
                window_queue.push_back(qi);

            int ntoerase = window_queue.size() - window_size;
            if(ntoerase > 0)
                window_queue.erase(window_queue.begin(), window_queue.begin()+ntoerase);
            send_update = true;
            //while(window_queue.size() > window_size)
            //    window_queue.pop_front();
            m.unlock();
        }
    }
};

int main(int argc, char * argv[])
{
    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.configure( argc, argv );

    /* create the module */
    e2vidPacker instance;
    return instance.runModule(rf);
}
