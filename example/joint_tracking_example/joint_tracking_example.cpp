#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/cv/Cv.h>
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
    deque<AE> evs_queue, evsFullImg;
    deque<sklt> pose2img;
    std::mutex m;
    sklt pose, dpose, poseGT;
    hpecore::jointMotionEstimator tracker;
    roiq qROI;
    int roiWidth = 20;
    int roiHeight = roiWidth;
    skltJoint jointName;
    string jointNameStr;
    int dimY, dimX;
    int nevs = 0;
    bool plotCv = false;
    cv::Mat fullImg;
    bool initTimer = false;
    double avgF = 0;
    BufferedPort<ImageOf<PixelRgb> > image_port;

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

        if(!image_port.open(getName("/img:o"))) {
            yError() << "Could not open output port";
            return false;
        }

        // connect ports
        yarp.connect("/file/ch3dvs:o", getName("/AE:i"), "fast_tcp");
        yarp.connect("/file/ch3GT10Hzskeleton:o", getName("/SKLT:i"), "fast_tcp");
        yarp.connect(getName("/img:o"), "/yarpview/img:i", "fast_tcp");

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


        // intialize velocities
        for (size_t i = 0; i < dpose.size(); i++)
        {
            dpose[i].u = 0;
            dpose[i].v = 0;
        }

        
        if(rf.check("cv"))
            plotCv = true;
        dimX = rf.check("dimX", Value(346)).asInt();
        dimY = rf.check("dimX", Value(240)).asInt();
        yInfo() << "Image dimensions = [" << dimX << ", " << dimY << "]";
        fullImg = cv::Mat::zeros(cv::Size(346, 240), CV_32F);
        cvtColor(fullImg, fullImg, cv::COLOR_GRAY2RGB);
      

        if(plotCv)
        {
            cv::namedWindow("HPE OUTPUT", cv::WINDOW_NORMAL);
            cv::resizeWindow("HPE OUTPUT", 346, 260);
        }
        

        //start the asynchronous and synchronous threads
        return Thread::start();
    }

    virtual double getPeriod()
    {
        return 0.04; //period of synchrnous thread
    }

    bool interruptModule()
    {
        //if the module is asked to stop ask the asynchrnous thread to stop
        input_port.interrupt();
        input_sklt.interrupt();
        image_port.interrupt();
        std::cout << "\033c";
        return Thread::stop();
    }

    void onStop()
    {
        //when the asynchrnous thread is asked to stop, close ports and do
        //other clean up
        input_port.close();
        input_sklt.close();
        image_port.close();
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
        if(plotCv)
        {
            if(initTimer) drawSkeleton(poseGT);
            cv::putText(fullImg, 
                        "Detection",
                        cv::Point(285, 210), // Coordinates
                        cv::FONT_HERSHEY_SIMPLEX, // Font
                        0.35, // Scale
                        cv::Scalar(0.0, 0.0, 0.8), // BGR Color
                        1, // Line Thickness 
                        cv:: LINE_AA); // Anti-alias 
            cv::putText(fullImg, 
                        "Tracking",
                        cv::Point(285, 225), // Coordinates
                        cv::FONT_HERSHEY_SIMPLEX, // Font
                        0.35, // Scale
                        cv::Scalar(0.8, 0.0, 0.0), // BGR Color
                        1, // Line Thickness
                        cv:: LINE_AA); // Anti-alias
            std::string strF = std::to_string(int(avgF));
            cv::putText(fullImg, 
                        "Freq = " + strF + "Hz",
                        cv::Point(20, 225), // Coordinates
                        cv::FONT_HERSHEY_SIMPLEX, // Font
                        0.35, // Scale
                        cv::Scalar(0.8, 0.8, 0.8), // BGR Color
                        1, // Line Thickness
                        cv:: LINE_AA); // Anti-alias
            cv::putText(fullImg, 
                        "HPE-core EDPR",
                        cv::Point(125, 20), // Coordinates
                        cv::FONT_HERSHEY_SIMPLEX, // Font
                        0.5, // Scale
                        cv::Scalar(0.8, 0.8, 0.8), // BGR Color
                        1, // Line Thickness 
                        cv:: LINE_AA); // Anti-alias 
            

            // plot tracked joint
            while(!pose2img.empty())
            {
                int x = pose2img.front()[jointName].u;
                int y = pose2img.front()[jointName].v;
                cv::Point pt(x, y);
                cv::drawMarker(fullImg, pt, cv::Scalar(0.8, 0, 0), 0, 4);
                pose2img.pop_front();
            }
            cv::imshow("HPE OUTPUT", fullImg);
            // output image using yarp
            fullImg *= 255;
            cv::Mat img_out;
            fullImg.convertTo(img_out, CV_8UC3);
            image_port.prepare().copy(yarp::cv::fromCvMat<PixelRgb>(img_out));
            image_port.write();
            cv::waitKey(1);
            fullImg = cv::Vec3f(0.4, 0.4, 0.4);
        }
    
        return Thread::isRunning();
    }


    void drawSkeleton(sklt poseGT)
    {
        // plot detected joints
        for(int i=0; i<13; i++)
        {
            int x = int(poseGT[i].u);
            int y = int(poseGT[i].v);
            cv::Point pt(x, y);
            cv::drawMarker(fullImg, pt, cv::Scalar(0, 0, 0.8), 1, 4);
            
        }
        // plot links between joints
        cv::Point head(int(poseGT[0].u), int(poseGT[0].v));
        cv::Point shoulderR(int(poseGT[1].u), int(poseGT[1].v));
        cv::Point shoulderL(int(poseGT[2].u), int(poseGT[2].v));
        cv::Point elbowR(int(poseGT[3].u), int(poseGT[3].v));
        cv::Point elbowL(int(poseGT[4].u), int(poseGT[4].v));
        cv::Point hipL(int(poseGT[5].u), int(poseGT[5].v));
        cv::Point hipR(int(poseGT[6].u), int(poseGT[6].v));
        cv::Point handR(int(poseGT[7].u), int(poseGT[7].v));
        cv::Point handL(int(poseGT[8].u), int(poseGT[8].v));
        cv::Point kneeR(int(poseGT[9].u), int(poseGT[9].v));
        cv::Point kneeL(int(poseGT[10].u), int(poseGT[10].v));
        cv::Point footR(int(poseGT[11].u), int(poseGT[11].v));
        cv::Point footL(int(poseGT[12].u), int(poseGT[12].v));

        cv::Scalar colorS = cv::Scalar(0, 0, 0.5);
        int th = 1;
        cv::line(fullImg, head, shoulderL, colorS, th);
        cv::line(fullImg, head, shoulderR, colorS, th);
        cv::line(fullImg, shoulderL, shoulderR, colorS, th);
        cv::line(fullImg, shoulderL, elbowL, colorS, th);
        cv::line(fullImg, shoulderR, elbowR, colorS, th);
        cv::line(fullImg, shoulderL, hipL, colorS, th);
        cv::line(fullImg, shoulderR, hipR, colorS, th);
        cv::line(fullImg, hipL, hipR, colorS, th);
        cv::line(fullImg, elbowL, handL, colorS, th);
        cv::line(fullImg, elbowR, handR, colorS, th);
        cv::line(fullImg, hipL, kneeL, colorS, th);
        cv::line(fullImg, hipR, kneeR, colorS, th);
        cv::line(fullImg, kneeR, footR, colorS, th);
        cv::line(fullImg, kneeL, footL, colorS, th);
    }


    void evsToImage(deque<AE> &evs)
    {
        while(!evs.empty())
        {
            int x = evs.front().x;
            int y = evs.front().y;
            int p = evs.front().polarity;
            if(x>=0 && x< dimX && y>=0 && y<dimY)
            {
                if(p)
                    fullImg.at<cv::Vec3f>(y, x) = cv::Vec3f(1.0, 1.0, 1.0);
                else
                    fullImg.at<cv::Vec3f>(y, x) = cv::Vec3f(0.0, 0.0, 0.0);
            }
            evs.pop_front();
        }
    }

    //asynchronous thread run forever
    void run()
    {
        Stamp evStamp, skltStamp;
        Bottle *bot_sklt;
        double skltTs;
        const vector<AE> *q;
        double t0, t1=0, tprev, t2 = 0;
        long int mes = 0;
        double freq = 0;
        
        
        while (true)
        {
            double dt = 0.0;
            t1 = Time::now() - t0;
            // read detections
            // int N = input_sklt.getPendingReads();
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
                poseGT = pose;
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
                avgF = freq/mes; // average freq
                if(mes>100)
                {
                    mes= 0;
                    freq=0;
                }
                // avgF = 1/(t2-tprev); // instant freq
                // yInfo() << "\033c" << avgF;
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
                    evsFullImg.push_back(qi);
                    nevs++;
                }
            }
            qROI.setSize(int((qROI.roi[1] - qROI.roi[0]) * (qROI.roi[3] - qROI.roi[2])/5));
            
            // Add events to output image
            if(initTimer) evsToImage(evsFullImg); 

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
                        dpose = tracker.estimateVelocity(evs, evsTs, jointName, nevs/4, dpose);  // get veocities from delta ts
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
                    pose2img.push_back(pose);
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
