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
#include "matplotlibcpp.h"

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;
using namespace hpecore;
namespace plt = matplotlibcpp;
using std::tuple;
using yarp::os::Bottle;
using yarp::os::BufferedPort;
using skeleton = std::vector<std::tuple<double, double>>;


class jointTrack : public RFModule, public Thread
{

private:
    vReadPort<vector<AE>> input_port;
    BufferedPort<Bottle> input_sklt;
    std::ofstream output_writer, aux_out, vel_out;
    deque<AE> evs_queue;
    std::mutex m;
    // skeleton pose, dpose;
    sklt pose, dpose;
    hpecore::jointMotionEstimator tracker;
    roiq qROI;
    int roiWidth = 20;
    int roiHeight = roiWidth;
    skltJoint jointName;
    string jointNameStr;
    // cv::Mat projImg = cv::Mat::zeros(cv::Size(346, 240), CV_32FC1);
    // cv::Mat projImg = cv::Mat::zeros(cv::Size(346, 240), CV_32FC1);
    cv::Mat projNew = cv::Mat::zeros(cv::Size(346, 240), CV_32F);
    cv::Mat projPrev = cv::Mat::zeros(cv::Size(346, 240), CV_32F);
    // cv::Mat diff = cv::Mat::zeros(cv::Size(346, 240), CV_32F);
    // cv::Mat projImg = cv::Mat::zeros(cv::Size(346, 240), CV_32F);
    cv::Mat matTe = cv::Mat::zeros(cv::Size(346, 240), CV_32FC1);
    float corr;
    int dimY = 240, dimX = 346;
    double** Te = new double*[dimY];
    bool plotPython = false, cvPlot = false;
    std::vector< double > xJ, yJ, tJ, xGT, yGT, tGT;
    int nevs = 0;
 
    
    // // float** Te = new float[240][346];

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

        aux_out.open("aux_out.txt");
        if (!aux_out.is_open())
            yError() << "Could not open pose writer!!";
        else
            yInfo() << "aux_out writer opened";

        vel_out.open("vel_out.txt");
        if (!vel_out.is_open())
            yError() << "Could not open pose writer!!";
        else
            yInfo() << "vel_out writer opened";

        

        jointNameStr = rf.check("joint", Value("handL")).asString();
        jointName = str2enum(jointNameStr);

        if(rf.check("plot"))
            plotPython = true;

        yInfo() << "plotPython = " << plotPython;

        if(rf.check("cv"))
            cvPlot = true;
        yInfo() << "cvPlot = " << cvPlot;
        
        if(cvPlot)
        {
            cv::namedWindow("NEW EVENTS", cv::WINDOW_NORMAL);
            cv::resizeWindow("NEW EVENTS", 346, 260);

            cv::namedWindow("OLD EVENTS", cv::WINDOW_NORMAL);
            cv::resizeWindow("OLD EVENTS", 346, 260);

            // cv::namedWindow("DEBUG3", cv::WINDOW_NORMAL);
            // cv::resizeWindow("DEBUG3", 346, 260);
        }

        // intialize velocities
        for (size_t i = 0; i < dpose.size(); i++)
        {
            dpose[i].u = 0;
            dpose[i].v = 0;
        }

        // Expected times
        for (int i = 0; i < dimY; i++)
            Te[i] = new double[dimX];

        for (int i = 0; i < dimY; i++) 
        {
            for (int j = 0; j < dimX; j++)
            {
                Te[i][j] = 1e10;
            }
            // std::cout << std::endl;
        }

        // cvtColor(projImg, projImg, cv::COLOR_GRAY2RGB);


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
        vel_out.close();
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

    float correlation(cv::Mat &image1, cv::Mat &image2)
    {
        cv::Mat corr;
        cv::matchTemplate(image1, image2, corr, cv::TM_CCORR_NORMED);
        return corr.at<float>(0,0);
    }

    //synchronous thread
    virtual bool updateModule()
    {

        
        // if(qROI.q.size())
        // {
        //     cv::Mat test = cv::Mat::zeros(cv::Size(346, 240), CV_32FC1);
        //     test += 2;
        //     test *= 0.2;
        //     for(auto v = qROI.q.rbegin(); v < qROI.q.rend(); v++)
        //     {
        //         int value = v->polarity;
        //         if(v->x<346 && v->y<240)
        //         {
        //             if(value > 0)
        //                 test.at<float>(v->y, v->x) = 0.0f;
        //             if(value <= 0)
        //                 test.at<float>(v->y, v->x) = 1.0f;
        //         }
        //     }
        //     // for(int i=qROI.roi[0]; i<=qROI.roi[1]; i++)
        //     // {
        //     //     test.at<float>(qROI.roi[2], i) = 0.7f;
        //     //     test.at<float>(qROI.roi[3], i) = 0.7f;
        //     // }
        //     // for(int i=qROI.roi[2]; i<=qROI.roi[3]; i++)
        //     // {
        //     //     test.at<float>(i, qROI.roi[0]) = 0.7f;
        //     //     test.at<float>(i, qROI.roi[1]) = 0.7f;
        //     // }
        //     cv::Mat frame;
        //     // test.convertTo(frame, CV_8UC3, 255);
        //     cvtColor(test, frame, cv::COLOR_GRAY2RGB);

        //     int x = (qROI.roi[0] + qROI.roi[1])/2;
        //     int y = (qROI.roi[2] + qROI.roi[3])/2;
        //     double vx = int(dpose[jointName].u);
        //     double vy = int(dpose[jointName].v);
        //     cv::Point pt1(x, y);
        //     int k = 2;
        //     cv::Point pt2(x+int(vx/k), y+int(vy/k));
        //     // cv.arrowedLine(test, pt1, pt2, color[, thickness[, line_type[, shift[, tipLength]]]]	) ->	img
            

        //     if(pt2.x < 0) pt2.x =0;
        //     if(pt2.x > 345) pt2.x =345;
        //     if(pt2.y < 0) pt2.y =0;
        //     if(pt2.y > 239) pt2.y =239;
            
        //     // test.convertTo(test, CV_32FC4, 1.0); 
        //     // if(x+int(vx/k)>0 && x+int(vx/k) < 346 && y+int(vy/k)>0 && y+int(vy/k)<240)
        //         cv::arrowedLine(frame, pt1, pt2, cv::Scalar(0, 0, 0.4), 2);


        //     int xC = int(pose[jointName].u);
        //     int yC = int(pose[jointName].v);
        //     cv::Point pt3(xC, yC);
        //     //tracked
        //     cv::drawMarker(frame, pt3, cv::Scalar(0.8, 0, 0), 0, 4);
        //     // ground-truth
        //     cv::drawMarker(frame, pt1, cv::Scalar(0, 0, 0.8), 1, 4);

        //     cv::Point ptA(qROI.roi[0], qROI.roi[2]);
        //     cv::Point ptB(qROI.roi[1], qROI.roi[3]);
        //     cv:rectangle(frame, ptA, ptB, cv::Scalar(0.7, 0.7, 0.7), 1, 16);

        //     cv::imshow("DEBUG", frame);


        //     double t1 = Time::now();
        //     // string fileName = "/projects/hpe-core/example/joint_tracking_example/build/imgs" + std::to_string(t1) + ".jpg";
        //     string fileName = "imgs/" + std::to_string(t1) + ".jpg";
            
        //     // cv::imwrite(fileName, test);  
        //     cv::waitKey(1);
        // }
        
        
        // corr = correlation(projImg, projImg2);
        // yInfo() << (1-corr)*100;
        // cv::imshow("NEW EVENTS", projNew);
        // cv::imshow("OLD EVENTS", projNew2);
        // cv::absdiff(projImg, projImg2, diff);
        // diff = projImg + projImg2;
        // // cv::subtract(projImg,projImg2,diff); 
        // // diff += 0.2;
        // // diff *= 0.2;
        // cv::imshow("DEBUG3", diff);
        // // cv::imshow("DEBUG3", (projImg-projImg2)*2);
        if(cvPlot)
        {
            cv::imshow("NEW EVENTS", projNew);
            cv::imshow("OLD EVENTS", matTe);
            cv::waitKey(1);
        }
        if(plotPython && nevs)
        {
            plt::clf();
            plt::ylim(0, 346);
            if(tJ.back() > 5)
                plt::xlim(int(tJ.back()-5), int(tJ.back())+1);
            else
                plt::xlim(0, 5);
            plt::plot(tJ, xJ, ".");
            plt::plot(tJ, yJ, ".");
            plt::plot(tGT, xGT, "--");
            plt::plot(tGT, yGT, "--");
            plt::grid(true);
            plt::title("Tracking of " + jointNameStr);
            plt::ylabel("x/y [px]");
            plt::xlabel("Time [sec]");
            plt::pause(0.001);
        }
        
        
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
                // skeleton builtPose = buildSkeleton(*sklt_lst); // using tuples
                sklt builtPose = buildSklt(*sklt_lst); // using array
                pose = tracker.resetPose(builtPose);   // reset pose
                // set roi for just one joint (left hand)
                int x = builtPose[jointName].u;
                int y = builtPose[jointName].v;
                qROI.setROI(x - roiWidth / 2, x + roiWidth / 2, y - roiHeight / 2, y + roiHeight / 2);
                aux_out << t1 << " ";
                for (auto &t : builtPose)
                    aux_out << t.u << " " << t.v << " ";
                aux_out << std::endl;
            }

            /*  EVENT BATCH */
            // read events
            int np = input_port.queryunprocessed();
            if(np)
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
                    // plt::figure_size(1920,1080);
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
            // if(nevs)
            //     yInfo() << nevs;
             /*  END EVENT BATCH */
            
            // /*  TIME WINDOW */
            // double period = 10e-3;
            // if(initTimer)
            // {
            //     if(1/dpose[jointName].u < period) period = 1/dpose[jointName].u;
            //     if(1/dpose[jointName].v < period) period = 1/dpose[jointName].v;    
            // }
            // int nevs = 0;
            // while(dt < update_time)
            // {
            //     //if we ran out of events get a new queue
            //     if(i >= q->size())
            //     {
            //         i = 0;
            //         // q = input_port.read(evStamp);
            //     }
            //     if(pt < 0) pt = q->front().stamp;
            //         qROI.add((*q)[i]);
            //     i++;
            //     nevs++;
            //     // if(nevs && !initTimer)
            //     // {
            //     //     initTimer = true;
            //     //     t0 = Time::now();
            //     // }
            //     if(qROI.q.size())
            //         dt = vtsHelper::deltaMS(qROI.q.front().stamp, pt);
            // }
            // // if(nevs)
            // // {
            // //     tprev = t2;
            // //     t2 = Time::now() - t0;
            // // }
            
            // // yInfo() << nevs;
            // pt = qROI.q.front().stamp;
            // /*  END TIME WINDOW */
            
            //  if(bot_sklt)
            //      yInfo() << "timer = " << t1 << "\t det = " << skltTs << "\t ev =" << evStamp.getTime();
            // yInfo() << "timer = " << t1 << "\t v = [" << dpose[jointName].u << ", " << dpose[jointName].v << "]";
            
            qROI.setSize(int((qROI.roi[1] - qROI.roi[0]) * (qROI.roi[3] - qROI.roi[2])/5));
            // qROI.setSize(nevs);

            // Process data for tracking
            if (pose.size()) // a pose has been detected before
            {
                if (nevs && qROI.q.size() && !bot_sklt)// && qROI.q.front().stamp * vtsHelper::tsscaler > skltTs) // there are events to process
                {
                    std::deque<joint> evs;
                    std::deque<double> evsTs;
                    std::deque<int> evsPol;
                    tracker.getEventsUV(qROI.q, evs, evsTs, vtsHelper::tsscaler, evsPol); // get events u,v coords

                    // Method 1: time diff on adjacent events 
                    // if(nevs > 75)
                    dpose = tracker.estimateVelocity(evs, evsTs, jointName, nevs/2, dpose);  // get veocities from delta ts
                    

                    // projNew = tracker.projNew(evs, evsTs, evsPol, jointName, nevs, dpose, t2);
                    // // projPrev = tracker.projPrev(evs, evsTs, evsPol, jointName, nevs, dpose, t2);
                    // projPrev = tracker.estimateFire(evs, evsTs, evsPol, jointName, nevs, pose, dpose, t2, Te, matTe);

                    // Method 2: expected fire time of events
                    if(cvPlot)
                    {
                        projNew = tracker.projNew(evs, evsTs, evsPol, jointName, nevs, dpose, t2);
                        projPrev = tracker.estimateFire(evs, evsTs, evsPol, jointName, nevs, pose, dpose, t2, Te, matTe);
                    }
                    
                    /*
                    estimate velocity based on image difference
                    if there is/isn't pixel wehere expected increase/decrease previous vel
                    */
                    // dpose = tracker.comparteProjs(jointName, dpose, projNew, projPrev);


                    // print_sklt(dpose);
                    double dt = (qROI.q.front().stamp - qROI.q.back().stamp) * vtsHelper::tsscaler;
                    double dt2 = t2 - tprev;
                    // if(nevs > 75)
                    tracker.fusion(&pose, dpose, dt); // should integrate from pose eith new velocity
                    // write integrated output to file
                    // output_writer << qROI.q.front().stamp * vtsHelper::tsscaler << " ";
                    output_writer << t1 << " ";
                    for (auto &t : pose)
                        output_writer << t.u << " " << t.v << " ";
                    output_writer << std::endl;
                    // vel_out << qROI.q.front().stamp * vtsHelper::tsscaler << " " << nevs << " " << qROI.q.size() << std::endl;
                    vel_out  << t1 << " " << corr << std::endl;
                    // update roi
                    int x = pose[jointName].u;
                    int y = pose[jointName].v;
                    if(plotPython && t1<100)
                    {
                        xJ.push_back(pose[jointName].u);
                        yJ.push_back(pose[jointName].v);
                        tJ.push_back(t1);
                    }
                    // qROI.setROI(x - roiWidth / 2, x + roiWidth / 2, y - roiHeight / 2, y + roiHeight / 2);
                    // qROI.setROI(x-roiWidth/2, x + roiWidth/2, y-roiHeight/2, y + roiHeight/2);
                    // qROI.setSize(0);
                }
                else if (bot_sklt) // there weren't events to process but a detection occured
                {
                    // write detected output to file
                    output_writer << t1 << " ";
                    for (auto &t : pose)
                        output_writer << t.u << " " << t.v << " ";
                    output_writer << std::endl;
                    if(plotPython && t1<100)
                    {
                        xGT.push_back(pose[jointName].u);
                        yGT.push_back(pose[jointName].v);
                        tGT.push_back(t1);
                    }
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
