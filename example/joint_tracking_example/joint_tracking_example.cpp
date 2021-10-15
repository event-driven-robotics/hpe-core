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

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;
using namespace hpecore;
using std::tuple;
using yarp::os::Bottle;
using yarp::os::BufferedPort;
using skeleton = std::vector<std::tuple<double, double>>;


class jointTrack : public RFModule, public Thread
{

private:
    vReadPort<vector<AE>> input_port;
    BufferedPort<Bottle> input_sklt;
    std::ofstream output_writer;
    deque<AE> evs_queue;
    std::mutex m;
    // skeleton pose, dpose;
    sklt pose, dpose;
    hpecore::jointMotionEstimator tracker;
    roiq qROI;
    int roiWidth = 20;
    int roiHeight = 20;
    skltJoint jointName;

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
        // yarp.connect("/file/ch3GTskeleton:o", getName("/SKLT:i"), "fast_tcp");

        output_writer.open("output.txt");
        if (!output_writer.is_open())
            yError() << "Could not open pose writer!!";
        else
            yInfo() << "Pose writer opened";

        cv::namedWindow("DEBUG", cv::WINDOW_NORMAL);
        cv::resizeWindow("DEBUG", 346, 260);

        string jointNameStr = rf.check("joint", Value("handL")).asString();
        jointName = str2enum(jointNameStr);

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
        // output_port.close();
    }

    skeleton buildSkeleton(Bottle &readBottle)
    {
        skeleton newPose;
        for (size_t i = 0; i < readBottle.size(); i = i + 2)
        {
            Value &x = readBottle.get(i);
            Value &y = readBottle.get(i + 1);
            newPose.emplace_back(std::make_tuple(x.asInt32(), y.asInt32()));
        }
        return newPose;
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

        // cv::Mat test = cv::Mat::zeros(cv::Size(346, 240), CV_32FC1);
        // test += 2;
        // test *= 0.2;
        // for(auto v = qROI.q.rbegin(); v < qROI.q.rend(); v++)
        // {
        //     int value = v->polarity;
        //     if(v->x<346 && v->y<240)
        //     {
        //         if(value > 0)
        //             test.at<float>(v->y, v->x) = 0.0f;
        //         if(value <= 0)
        //             test.at<float>(v->y, v->x) = 1.0f;
        //     }
        // }
        // if(qROI.q.size())
        // {
        //     // while(!evs_queue.empty())
        //     // {
        //     //     auto v = evs_queue.front();
        //     //     int value = v.polarity;
        //     //     if(v.x<346 && v.y<240)
        //     //     {
        //     //         if(value > 0)
        //     //         {
        //     //             test.at<float>(v.y, v.x) = 0.0f;
        //     //         }
        //     //         if(value <= 0)
        //     //         {
        //     //             test.at<float>(v.y, v.x) = 1.0f;
        //     //         }
        //     //     }  
        //     //     evs_queue.pop_front();
        //     // }
        //      for(int i=qROI.roi[0]; i<=qROI.roi[1]; i++)
        //     {
        //         test.at<float>(qROI.roi[2], i) = 0.7f;
        //         test.at<float>(qROI.roi[3], i) = 0.7f;
        //     }
        //     for(int i=qROI.roi[2]; i<=qROI.roi[3]; i++)
        //     {
        //         test.at<float>(i, qROI.roi[0]) = 0.7f;
        //         test.at<float>(i, qROI.roi[1]) = 0.7f;
        //     }
        // }
            
        // cv::Mat frame;
        // test.convertTo(frame, CV_8U, 255);
        // cv::imshow("DEBUG", test);
        // cv::waitKey(1);

        return Thread::isRunning();
    }

    //asynchronous thread run forever
    void run()
    {
        Stamp evStamp, skltStamp;
        Bottle *bot_sklt;
        double skltTs;
        const vector<AE> *q = input_port.read(evStamp);
        if (!q || Thread::isStopping())
            return;
        // int jointName = elbowL;

        while (true)
        {
            // read detections
            bot_sklt = input_sklt.read(false);
            input_sklt.getEnvelope(skltStamp);
            skltTs = skltStamp.getTime();
            if (bot_sklt) // there is a detection
            {
                // yInfo() << "\tSKLT @ " << skltTs;
                Value &coords = (*bot_sklt).get(1);
                Bottle *sklt_lst = coords.asList();
                // build skeleton from reading
                // skeleton builtPose = buildSkeleton(*sklt_lst); // using tuples
                sklt builtPose = buildSklt(*sklt_lst); // using array
                pose = tracker.resetPose(builtPose);   // reset pose
                // set roi for just one joint (left hand)
                int x = builtPose[jointName].u;
                int y = builtPose[jointName].v;
                qROI.setROI(x - roiWidth / 2, x + roiWidth / 2, y - roiHeight / 2, y + roiHeight / 2);
            }

            // read events
            int np = input_port.queryunprocessed();
            for (int i = 0; i < np; i++)
            {
                q = input_port.read(evStamp);
                if (!q || Thread::isStopping())
                    return;
                for (auto &qi : *q)
                {
                    qROI.add(qi);
                    // evs_queue.push_back(qi);
                }
            }
            qROI.setSize((qROI.roi[1] - qROI.roi[0]) * (qROI.roi[3] - qROI.roi[2]) / 3);

            // Process data for tracking
            if (pose.size()) // a pose has been detected before
            {
                if (np && qROI.q.size()) // there are events to process
                {
                    std::deque<joint> evs;
                    std::deque<double> evsTs;
                    tracker.getEventsUV(qROI.q, evs, evsTs, vtsHelper::tsscaler); // get events u,v coords
                    dpose = tracker.estimateVelocity(evs, evsTs, jointName);                 // get skeleton veocities
                    // print_sklt(dpose);
                    double dt = (qROI.q.front().stamp - qROI.q.back().stamp) * vtsHelper::tsscaler;
                    // yInfo() << dt;
                    // double dt = 1;
                    tracker.fusion(&pose, dpose, dt); // should integrate from pose eith new velocity
                    //write integrated output to file
                    output_writer << qROI.q.front().stamp * vtsHelper::tsscaler << " ";
                    for (auto &t : pose)
                        output_writer << t.u << " " << t.v << " ";
                    output_writer << std::endl;
                    // update roi
                    int x = pose[jointName].u;
                    int y = pose[jointName].v;
                    // qROI.setROI(x-roiWidth/2, x + roiWidth/2, y-roiHeight/2, y + roiHeight/2);
                    qROI.setSize((qROI.roi[1] - qROI.roi[0]) * (qROI.roi[3] - qROI.roi[2]) / 100);
                }
                else if (bot_sklt) // there weren't events to process but a detection occured
                {
                    //write detected output to file
                    output_writer << skltTs << " ";
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
