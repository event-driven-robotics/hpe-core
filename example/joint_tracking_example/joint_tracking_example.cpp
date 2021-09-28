#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <event-driven/all.h>
#include <mutex>
#include <hpe-core/utility.h>
#include <hpe-core/jointMotionEstimator.h>
#include "roiq.h"

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;
using namespace hpecore;
using yarp::os::BufferedPort;
using yarp::os::Bottle;
using std::tuple;
using skeleton = std::vector<std::tuple<double, double>>;

class jointTrack : public RFModule, public Thread {

private:
    vReadPort<vector<AE> > input_port;
    BufferedPort<Bottle> input_sklt;
    std::ofstream  output_writer;
    deque<AE> evs_queue;
    std::mutex m;
    skeleton pose, dpose;
    hpecore::jointMotionEstimator tracker;
    roiq qROI;
    int roiWidth = 5;
    int roiHeight = 5;
    
public:

    jointTrack() {}

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        //set the module name used to name ports
        setName((rf.check("name", Value("/jointTrack")).asString()).c_str());

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

        if(!input_sklt.open(getName("/SKLT:i"))) {
            yError() << "Could not open input port";
            return false;
        }

        // connect ports
        yarp.connect("/file/ch3dvs:o", getName("/AE:i"), "fast_tcp");
        yarp.connect("/file/ch3GT10Hzskeleton:o", getName("/SKLT:i"), "fast_tcp");
        // yarp.connect("/file/ch3GTskeleton:o", getName("/SKLT:i"), "fast_tcp");

        output_writer.open("output.txt");
        if(!output_writer.is_open())
            yError() << "Could not open pose writer!!";
        else
            yInfo() << "Pose writer opened";

        //start the asynchronous and synchronous threads
        return Thread::start();
    }

    virtual double getPeriod()
    {
        return 0; //period of synchrnous thread
    }

    bool interruptModule()
    {
        //if the module is asked to stop ask the asynchrnous thread to stop
        input_port.interrupt();
        input_sklt.interrupt();
        return Thread::stop();
    }

    void onStop()
    {
        //when the asynchrnous thread is asked to stop, close ports and do
        //other clean up
        input_port.close();
        input_sklt.close();
        output_writer.close();
        // output_port.close();
    }

    skeleton buildSkeleton(Bottle& readBottle)
    {
        skeleton newPose;
        for (size_t i = 0; i < readBottle.size(); i = i+2)
        {
            Value& x = readBottle.get(i);
            Value& y = readBottle.get(i+1);
            newPose.emplace_back(std::make_tuple(x.asInt32(), y.asInt32()));
        }
        return newPose;
    }

    //synchronous thread
    virtual bool updateModule()
    {
        
        return Thread::isRunning();
    }

    //asynchronous thread run forever
    void run() {
        Stamp evStamp, skltStamp;
        Bottle* bot_sklt;
        double skltTs;
        const vector<AE>* q = input_port.read(evStamp);
        if(!q || Thread::isStopping()) return;

        while (true)
        {
            bot_sklt = input_sklt.read(false); // read full skeleteon
            input_sklt.getEnvelope(skltStamp); 
            skltTs = skltStamp.getTime(); // get timestamp
            if(bot_sklt)
            {
                // yInfo() << "\tSKLT @ " << skltTs; 
                Value& coords = (*bot_sklt).get(1);
                Bottle* sklt_lst = coords.asList();
                skeleton builtPose = buildSkeleton(*sklt_lst); // build skeleton from reading
                pose = tracker.resetPose(builtPose); // reset pose
                int x = std::get<0>(builtPose.at(4));
                int y = std::get<1>(builtPose.at(4));
                qROI.setROI(x, x + roiWidth, y, y + roiHeight);
            }
            

            int np = input_port.queryunprocessed();
            for(int i = 0; i < np; i++)
            {
                q = input_port.read(evStamp);
                if(!q || Thread::isStopping()) return;
                for (auto& qi : *q)
                {
                    qROI.add(qi);
                }
            }
            qROI.setSize((qROI.roi[1]-qROI.roi[0])*(qROI.roi[3]-qROI.roi[2])/3);

            if(pose.size())
            {
                if(np && qROI.q.size())
                {
                    dpose = tracker.estimateVelocity(); // should return zero skeleton (zero veocities)
                    // dpose = tracker.estimateVelocity(qROI.q); // should return zero skeleton (zero veocities)
                    tracker.fusion(&pose, dpose); // should integrate from pose eith new velocity
                    //write output to file
                    output_writer << qROI.q.front().stamp* vtsHelper::tsscaler << " ";
                    for(auto &t : pose)
                        output_writer << std::get<0>(t) << " " << std::get<1>(t)<< " " ;
                    output_writer << std::endl;
                }
                else if(bot_sklt)
                {
                    output_writer << skltTs << " ";
                    for(auto &t : pose)
                        output_writer << std::get<0>(t) << " " << std::get<1>(t)<< " " ;
                    output_writer << std::endl;
                } 
            }
            
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
    jointTrack instance;
    return instance.runModule(rf);
}
