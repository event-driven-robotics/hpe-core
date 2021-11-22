#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <event-driven/all.h>
#include <mutex>
#include <hpe-core/utility.h>
#include <hpe-core/jointMotionEstimator.h>
#include "roiq.h"
#include <array>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;
using namespace hpecore;
using std::tuple;
using yarp::os::Bottle;
using yarp::os::BufferedPort;
// using std::tuple;
// using skeleton = std::vector<std::tuple<double, double>>;


class jointTrack : public RFModule, public Thread
{

private:
    vReadPort<vector<AE>> input_port;
    BufferedPort<Bottle> input_sklt;
    std::ofstream output_writer, aux_out, vel_out;
    deque<AE> evs_queue;
    std::mutex m;
    sklt pose, dpose;
    hpecore::jointMotionEstimator tracker;
    roiq qROI;
    int roiWidth = 20;
    int roiHeight = roiWidth;
    skltJoint jointName;
    string jointNameStr;
    int dimY = 240, dimX = 346;
    int nevs = 0;
 

public:
    jointTrack() {}

    virtual bool configure(yarp::os::ResourceFinder &rf)
    {
        std::cout << "\033c";
        yInfo() << "\t\t - = HPE - JOINT MOTION ESTIMATION = -";
        //set the module name used to name ports
        setName((rf.check("name", Value("/jointTrack")).asString()).c_str());

        /* initialize yarp network */
        yarp::os::Network yarp;
        if (!yarp.checkNetwork(2.0))
        {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }

        //open io ports
        if (!input_port.open(getName("/AE:i")))
        {
            yError() << "Could not open input port";
            return false;
        }

        if (!input_sklt.open(getName("/SKLT:i")))
        {
            yError() << "Could not open input port";
            return false;
        }

        // connect ports
        yarp.connect("/file/ch3dvs:o", getName("/AE:i"), "fast_tcp");
        yarp.connect("/file/ch3GT10Hzskeleton:o", getName("/SKLT:i"), "fast_tcp");

        output_writer.open("output.txt");
        if (!output_writer.is_open())
            yError() << "Could not open pose writer!!";
        else
            yInfo() << "Pose writer opened";

        aux_out.open("aux_out.txt");
        if (!aux_out.is_open())
            yError() << "Could not open pose writer!!";
        else
            yInfo() << "aux_out writer opened";
        
        jointNameStr = rf.check("joint", Value("handL")).asString();
        jointName = str2enum(jointNameStr);

        // if(rf.check("plot"))
        //     plotPython = true;
      
        // intialize velocities
        for (size_t i = 0; i < dpose.size(); i++)
        {
            dpose[i].u = 0;
            dpose[i].v = 0;
        }

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
        std::cout << "\033c";
        return Thread::stop();
    }

    void onStop()
    {
        //when the asynchrnous thread is asked to stop, close ports and do
        //other clean up
        input_port.close();
        input_sklt.close();
        output_writer.close();
        aux_out.close();
        // output_port.close();
    }

    sklt buildSklt(Bottle &readBottle)
    {
        sklt newPose;
        for (size_t i = 0; i < readBottle.size(); i = i + 2)
        {
            Value &u = readBottle.get(i);
            Value &v = readBottle.get(i + 1);
            newPose[i / 2].u = u.asInt32();
            newPose[i / 2].v = v.asInt32();
        }
        return newPose;
    }

    //synchronous thread
    virtual bool updateModule()
    {

        return Thread::isRunning();
    }

    //asynchronous thread run forever
    void run()
    {
        Stamp evStamp, skltStamp;
        Bottle *bot_sklt;
        double skltTs;
        const vector<AE> *q;
        bool initTimer = false;
        double t0, t1=0, tprev, t2 = 0;
        int pt = -1;
        double update_time = 1.0;
        int i=0;
        long int mes = 0;
        double freq = 0;
        
        
        while (true)
        {
            double dt = 0.0;
            t1 = Time::now() - t0;
            // read detections
            int N = input_sklt.getPendingReads();
            bot_sklt = input_sklt.read(false);
            input_sklt.getEnvelope(skltStamp);
            skltTs = skltStamp.getTime();
            if(bot_sklt) // there is a detection
            {
                // yInfo() << "\tSKLT @ " << skltTs;
                Value &coords = (*bot_sklt).get(1);
                Bottle *sklt_lst = coords.asList();
                // build skeleton from reading
                sklt builtPose = buildSklt(*sklt_lst);
                pose = tracker.resetPose(builtPose);   // reset pose
                // set roi for just one joint
                int x = builtPose[jointName].u;
                int y = builtPose[jointName].v;
                qROI.setROI(x - roiWidth / 2, x + roiWidth / 2, y - roiHeight / 2, y + roiHeight / 2);
                aux_out << t1 << " ";
                for (auto &t : builtPose)
                    aux_out << t.u << " " << t.v << " ";
                aux_out << std::endl;
            }

            // read events
            int np = input_port.queryunprocessed();
            if(np && initTimer)
            {
                tprev = t2;
                t2 = Time::now() - t0;
                freq += 1/(t2-tprev);
                mes++;
                double avg = freq/mes;
                yInfo() << "\033c" << avg;
            }
            nevs = 0;
            for (int i = 0; i < np; i++)
            {   
                if(!initTimer)
                {
                    initTimer = true;
                    t0 = Time::now();
                }
                q = input_port.read(evStamp);
                if (!q || Thread::isStopping())
                    return;
                for (auto &qi : *q)
                {
                    qROI.add(qi);
                    nevs++;
                }
            }
            qROI.setSize(int((qROI.roi[1] - qROI.roi[0]) * (qROI.roi[3] - qROI.roi[2])/5));
            
            // Process data for tracking
            if(pose.size()) // a pose has been detected before
            {
                if (nevs && qROI.q.size() && !bot_sklt)// && qROI.q.front().stamp * vtsHelper::tsscaler > skltTs) // there are events to process
                {
                    std::deque<joint> evs;
                    std::deque<double> evsTs;
                    std::deque<int> evsPol;
                    tracker.getEventsUV(qROI.q, evs, evsTs, vtsHelper::tsscaler, evsPol); // get events u,v coords
                    // Velocity estimation Method 1: time diff on adjacent events 
                    if(nevs > 20)
                        dpose = tracker.estimateVelocity(evs, evsTs, jointName, nevs/2, dpose);  // get veocities from delta ts
                    double dt = (qROI.q.front().stamp - qROI.q.back().stamp) * vtsHelper::tsscaler;
                    if(nevs > 20)
                        tracker.fusion(&pose, dpose, dt); // should integrate from pose eith new velocity
                    // write integrated output to file
                    // output_writer << qROI.q.front().stamp * vtsHelper::tsscaler << " ";
                    output_writer << t1 << " ";
                    for (auto &t : pose)
                        output_writer << t.u << " " << t.v << " ";
                    output_writer << std::endl;
                    // update roi
                    int x = pose[jointName].u;
                    int y = pose[jointName].v;
                    // qROI.setROI(x - roiWidth / 2, x + roiWidth / 2, y - roiHeight / 2, y + roiHeight / 2);
                }
                else if (bot_sklt) // there weren't events to process but a detection occured
                {
                    // write detected output to file
                    output_writer << t1 << " ";
                    for (auto &t : pose)
                        output_writer << t.u << " " << t.v << " ";
                    output_writer << std::endl;
                }
                
            }
        }
    }
};

int main(int argc, char *argv[])
{
    /* initialize yarp network */
    yarp::os::Network yarp;
    if (!yarp.checkNetwork(2))
    {
        std::cout << "Could not connect to YARP" << std::endl;
        return false;
    }
    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.configure(argc, argv);
    /* create the module */
    jointTrack instance;
    return instance.runModule(rf);
}
