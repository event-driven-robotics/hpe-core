#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/cv/Cv.h>
#include <event-driven/all.h>
#include <mutex>
#include <hpe-core/utility.h>
#include <hpe-core/jointMotionEstimator.h>
#include <hpe-core/representations.h>
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

#define N 13
typedef std::array<roiq, N> roiqSklt;


class jointTrack : public RFModule, public Thread
{

private:
    vReadPort<vector<AE>> input_port;
    BufferedPort<Bottle> input_sklt;
    std::ofstream output_writer, aux_out, vel_out;
    deque<AE> evsFullImg;
    deque<joint> pose2img[N];
    deque<joint> vels;
    std::mutex m;
    skeleton13 pose, dpose, poseGT;
    hpecore::jointMotionEstimator tracker;
    // roiqSklt qROI;
    roiq qROI[N];
    int roiWidth = 32;
    int roiHeight = roiWidth;
    jointName jointNum;
    string jointNameStr;
    int dimY, dimX;
    int nevs[N] = {0};
    bool plotCv = false;
    cv::Mat fullImg;
    bool initTimer = false;
    double avgF = 0;
    BufferedPort<ImageOf<PixelRgb> > image_port;
    float displayF = 25.0; // display frequency in Hz
    bool showF = false;
    cv::Mat SAE, SAE_vis;
    double** Te;
    int method;
    bool avgV = false;
    int detF;
    bool objectTracking = false;
    bool h36 = false;
    bool pastSurf = false;
    bool roi = false;
    cv::Point ptJoint;
    hpecore::surface S;

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

        detF = rf.check("detF", Value(10)).asInt();
        yInfo() << "Detections frequency = " << detF << " Hz";
        if(rf.check("obj"))
            objectTracking = true;

        

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
        jointNum = str2enum(jointNameStr);
        displayF = rf.check("F", Value(25)).asFloat32();
        method = rf.check("M", Value(2)).asInt();

        if(rf.check("avgV"))
            avgV = true;

        if(rf.check("roi"))
            roi = true;


        yInfo() << "Method for velocity estimations = " << method;

        // intialize velocities
        for (size_t i = 0; i < dpose.size(); i++)
        {
            dpose[i].u = 0;
            dpose[i].v = 0;
        }

        if(rf.check("cv"))
            plotCv = true;
        dimX = rf.check("dimX", Value(346)).asInt();
        dimY = rf.check("dimY", Value(260)).asInt();
        if(rf.check("h36"))
            h36 = true;

        // connect ports
        if(!objectTracking)
        {
            if(!h36) // dhp19 dataset
            {
                yarp.connect("/file/ch3dvs:o", getName("/AE:i"), "fast_tcp");
                // yarp.connect("/file/ch3GT10Hzskeleton:o", getName("/SKLT:i"), "fast_tcp");
                yarp.connect("/file/ch3GTskeleton:o", getName("/SKLT:i"), "fast_tcp");
            }
            else // H 3.6 dataset
            {
                yarp.connect("/file/ch0dvs:o", getName("/AE:i"), "fast_tcp");
                yarp.connect("/file/ch0GT50Hzskeleton:o", getName("/SKLT:i"), "fast_tcp");
                roiWidth = 20;
                roiHeight = roiWidth;
            }
        }
        else
        {
            yarp.connect("/file/leftdvs:o", getName("/AE:i"), "fast_tcp");
            yarp.connect("/file/502Hz:o", getName("/SKLT:i"), "fast_tcp");
            roiWidth = 60;
            roiHeight = 100;
            dimX = 640;
            dimY = 480;
            jointNum = str2enum("head");
        }

        yarp.connect(getName("/img:o"), "/yarpview/img:i", "fast_tcp");


        yInfo() << "Image dimensions = [" << dimX << ", " << dimY << "]";
        fullImg = cv::Mat::zeros(cv::Size(dimX, dimY), CV_32F);
        cvtColor(fullImg, fullImg, cv::COLOR_GRAY2RGB);
        if(rf.check("showF"))
            showF = true;

        if(rf.check("past"))
            pastSurf = true;

        if(plotCv)
        {
            int k = 1;
            if(!objectTracking) k =2;
            int x0 = 1600, y0 = 0;
            cv::namedWindow("HPE OUTPUT", cv::WINDOW_NORMAL);
            cv::resizeWindow("HPE OUTPUT", 640, 480);
            cv::moveWindow("HPE OUTPUT", x0+1920/3, y0);

            if(pastSurf)
            {
                cv::namedWindow("TOS", cv::WINDOW_NORMAL);
                cv::resizeWindow("TOS", 640, 480);
                cv::moveWindow("TOS", x0+1920/3-325, y0+ 600);
                

                cv::namedWindow("SAE", cv::WINDOW_NORMAL);
                cv::resizeWindow("SAE", 640, 480);
                cv::moveWindow("SAE", x0+1920/3+325,y0+ 600);
            }  
        }
        

        SAE = cv::Mat::zeros(cv::Size(dimX, dimY), CV_32F);
        SAE_vis = cv::Mat::zeros(cv::Size(dimX, dimY), CV_32F);
        // // Expected times
        // Te = new double*[dimY];
        // for (int i = 0; i < dimY; i++)
        //     Te[i] = new double[dimX];

        // for (int i = 0; i < dimY; i++) 
        // {
        //     for (int j = 0; j < dimX; j++)
        //     {
        //         Te[i][j] = 0;
        //     }
        // }


        // past events surface
        // S.getSurface();
        // S.init(dimX, dimY, 7, 0.3); // EROS
        S.init(dimX, dimY, 7, 2); // TOS


        //start the asynchronous and synchronous threads
        return Thread::start();
    }


    virtual double getPeriod()
    {
        return 1/displayF; //period of synchrnous thread
    }


    bool interruptModule()
    {
        //if the module is asked to stop ask the asynchrnous thread to stop
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


    skeleton13 buildSklt(Bottle &readBottle)
    {
        skeleton13 newPose;
        for (size_t i = 0; i < readBottle.size(); i = i + 2)
        {
            Value &u = readBottle.get(i);
            Value &v = readBottle.get(i + 1);
            newPose[i / 2].u = u.asInt32();
            newPose[i / 2].v = v.asInt32();
        }
        return newPose;
    }

void drawSkeleton(skeleton13 pose, cv::Mat img, cv::Scalar color)
    {
        // plot detected joints
        for(int i=0; i<13; i++)
        {
            int x = int(pose[i].u);
            int y = int(pose[i].v);
            cv::Point pt(x, y);
            if(x && y) cv::drawMarker(img, pt, color, 1, 8);
            
        }
        // plot links between joints
        cv::Point head(int(pose[0].u), int(pose[0].v));
        cv::Point shoulderR(int(pose[1].u), int(pose[1].v));
        cv::Point shoulderL(int(pose[2].u), int(pose[2].v));
        cv::Point elbowR(int(pose[3].u), int(pose[3].v));
        cv::Point elbowL(int(pose[4].u), int(pose[4].v));
        cv::Point hipL(int(pose[5].u), int(pose[5].v));
        cv::Point hipR(int(pose[6].u), int(pose[6].v));
        cv::Point handR(int(pose[7].u), int(pose[7].v));
        cv::Point handL(int(pose[8].u), int(pose[8].v));
        cv::Point kneeR(int(pose[9].u), int(pose[9].v));
        cv::Point kneeL(int(pose[10].u), int(pose[10].v));
        cv::Point footR(int(pose[11].u), int(pose[11].v));
        cv::Point footL(int(pose[12].u), int(pose[12].v));

        int th = 1;
        if(head.x && head.y && shoulderL.x && shoulderL.y) cv::line(img, head, shoulderL, color, th);
        if(head.x && head.y && shoulderR.x && shoulderR.y) cv::line(img, head, shoulderR, color, th);
        if(shoulderL.x && shoulderL.y && shoulderR.x && shoulderR.y) cv::line(img, shoulderL, shoulderR, color, th);
        if(shoulderL.x && shoulderL.y && elbowL.x && elbowL.y) cv::line(img, shoulderL, elbowL, color, th);
        if(shoulderR.x && shoulderR.y && elbowR.x && elbowR.y) cv::line(img, shoulderR, elbowR, color, th);
        if(shoulderL.x && shoulderL.y && hipL.x && hipL.y) cv::line(img, shoulderL, hipL, color, th);
        if(shoulderR.x && shoulderR.y && hipR.x && hipR.y) cv::line(img, shoulderR, hipR, color, th);
        if(hipL.x && hipL.y && hipR.x && hipR.y) cv::line(img, hipL, hipR, color, th);
        if(elbowL.x && elbowL.y && handL.x && handL.y) cv::line(img, elbowL, handL, color, th);
        if(elbowR.x && elbowR.y && handR.x && handR.y) cv::line(img, elbowR, handR, color, th);
        if(hipL.x && hipL.y && kneeL.x && kneeL.y) cv::line(img, hipL, kneeL, color, th);
        if(hipR.x && hipR.y && kneeR.x && kneeR.y) cv::line(img, hipR, kneeR, color, th);
        if(kneeR.x && kneeR.y && footR.x && footR.y) cv::line(img, kneeR, footR, color, th);
        if(kneeL.x && kneeL.y && footL.x && footL.y) cv::line(img, kneeL, footL, color, th);
    }

void drawRoi(cv::Mat img, cv::Scalar color, int th)
{
        // draw Roi rectangles
        if(ptJoint.x && ptJoint.y)
        {
            for(int i=0; i<N; i++)
            {
                cv::Point roi02(int(qROI[i].roi[0]), int(qROI[i].roi[2]));
                cv::Point roi12(int(qROI[i].roi[1]), int(qROI[i].roi[2]));
                cv::Point roi03(int(qROI[i].roi[0]), int(qROI[i].roi[3]));
                cv::Point roi13(int(qROI[i].roi[1]), int(qROI[i].roi[3]));
                cv::line(img, roi02, roi12, color, th);
                cv::line(img, roi03, roi13, color, th);
                cv::line(img, roi02, roi03, color, th);
                cv::line(img, roi12, roi13, color, th);
            }
        }           

}


    void drawSkeleton(skeleton13 poseGT)
    {
        // plot detected joints
        for(int i=0; i<13; i++)
        {
            int x = int(poseGT[i].u);
            int y = int(poseGT[i].v);
            cv::Point pt(x, y);
            if(x && y) cv::drawMarker(fullImg, pt, cv::Scalar(0, 0, 0.8), 1, 8);
            
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
        if(head.x && head.y && shoulderL.x && shoulderL.y) cv::line(fullImg, head, shoulderL, colorS, th);
        if(head.x && head.y && shoulderR.x && shoulderR.y) cv::line(fullImg, head, shoulderR, colorS, th);
        if(shoulderL.x && shoulderL.y && shoulderR.x && shoulderR.y) cv::line(fullImg, shoulderL, shoulderR, colorS, th);
        if(shoulderL.x && shoulderL.y && elbowL.x && elbowL.y) cv::line(fullImg, shoulderL, elbowL, colorS, th);
        if(shoulderR.x && shoulderR.y && elbowR.x && elbowR.y) cv::line(fullImg, shoulderR, elbowR, colorS, th);
        if(shoulderL.x && shoulderL.y && hipL.x && hipL.y) cv::line(fullImg, shoulderL, hipL, colorS, th);
        if(shoulderR.x && shoulderR.y && hipR.x && hipR.y) cv::line(fullImg, shoulderR, hipR, colorS, th);
        if(hipL.x && hipL.y && hipR.x && hipR.y) cv::line(fullImg, hipL, hipR, colorS, th);
        if(elbowL.x && elbowL.y && handL.x && handL.y) cv::line(fullImg, elbowL, handL, colorS, th);
        if(elbowR.x && elbowR.y && handR.x && handR.y) cv::line(fullImg, elbowR, handR, colorS, th);
        if(hipL.x && hipL.y && kneeL.x && kneeL.y) cv::line(fullImg, hipL, kneeL, colorS, th);
        if(hipR.x && hipR.y && kneeR.x && kneeR.y) cv::line(fullImg, hipR, kneeR, colorS, th);
        if(kneeR.x && kneeR.y && footR.x && footR.y) cv::line(fullImg, kneeR, footR, colorS, th);
        if(kneeL.x && kneeL.y && footL.x && footL.y) cv::line(fullImg, kneeL, footL, colorS, th);

        // draw Roi detection rectangle
        if(ptJoint.x && ptJoint.y && false)
        {
            cv::Point roi02(ptJoint.x-roiWidth/2, ptJoint.y-roiHeight/2);
            cv::Point roi12(ptJoint.x+roiWidth/2, ptJoint.y-roiHeight/2);
            cv::Point roi03(ptJoint.x-roiWidth/2, ptJoint.y+roiHeight/2);
            cv::Point roi13(ptJoint.x+roiWidth/2, ptJoint.y+roiHeight/2);
            cv::line(fullImg, roi02, roi12, cv::Scalar(0, 0, 0.25), 1);
            cv::line(fullImg, roi03, roi13, cv::Scalar(0, 0, 0.25), 1);
            cv::line(fullImg, roi02, roi03, cv::Scalar(0, 0, 0.25), 1);
            cv::line(fullImg, roi12, roi13, cv::Scalar(0, 0, 0.25), 1);
        }


        // draw Roi rectangle
        if(ptJoint.x && ptJoint.y && false)
        {
            for(int i=0; i<N; i++)
            {
                cv::Point roi02(int(qROI[i].roi[0]), int(qROI[i].roi[2]));
                cv::Point roi12(int(qROI[i].roi[1]), int(qROI[i].roi[2]));
                cv::Point roi03(int(qROI[i].roi[0]), int(qROI[i].roi[3]));
                cv::Point roi13(int(qROI[i].roi[1]), int(qROI[i].roi[3]));
                cv::line(fullImg, roi02, roi12, cv::Scalar(0, 0.5, 0), 2);
                cv::line(fullImg, roi03, roi13, cv::Scalar(0, 0.5, 0), 2);
                cv::line(fullImg, roi02, roi03, cv::Scalar(0, 0.5, 0), 2);
                cv::line(fullImg, roi12, roi13, cv::Scalar(0, 0.5, 0), 2);
            }
        }

        // draw tracked skeleton
        colorS = cv::Scalar(0.5, 0, 0);
        th = 1;

        cv::Point head2(int(pose[0].u), int(pose[0].v));
        cv::Point shoulderR2(int(pose[1].u), int(pose[1].v));
        cv::Point shoulderL2(int(pose[2].u), int(pose[2].v));


        if(pose[0].u && pose[0].v && pose[2].u && pose[2].v) cv::line(fullImg, head2, shoulderL2, colorS, th);
        // if(head.x && head.y && shoulderR.x && shoulderR.y) cv::line(fullImg, head, shoulderR, colorS, th);
        // if(shoulderL.x && shoulderL.y && shoulderR.x && shoulderR.y) cv::line(fullImg, shoulderL, shoulderR, colorS, th);
        // if(shoulderL.x && shoulderL.y && elbowL.x && elbowL.y) cv::line(fullImg, shoulderL, elbowL, colorS, th);
        // if(shoulderR.x && shoulderR.y && elbowR.x && elbowR.y) cv::line(fullImg, shoulderR, elbowR, colorS, th);
        // if(shoulderL.x && shoulderL.y && hipL.x && hipL.y) cv::line(fullImg, shoulderL, hipL, colorS, th);
        // if(shoulderR.x && shoulderR.y && hipR.x && hipR.y) cv::line(fullImg, shoulderR, hipR, colorS, th);
        // if(hipL.x && hipL.y && hipR.x && hipR.y) cv::line(fullImg, hipL, hipR, colorS, th);
        // if(elbowL.x && elbowL.y && handL.x && handL.y) cv::line(fullImg, elbowL, handL, colorS, th);
        // if(elbowR.x && elbowR.y && handR.x && handR.y) cv::line(fullImg, elbowR, handR, colorS, th);
        // if(hipL.x && hipL.y && kneeL.x && kneeL.y) cv::line(fullImg, hipL, kneeL, colorS, th);
        // if(hipR.x && hipR.y && kneeR.x && kneeR.y) cv::line(fullImg, hipR, kneeR, colorS, th);
        // if(kneeR.x && kneeR.y && footR.x && footR.y) cv::line(fullImg, kneeR, footR, colorS, th);
        // if(kneeL.x && kneeL.y && footL.x && footL.y) cv::line(fullImg, kneeL, footL, colorS, th);

        
    }

    //synchronous thread
    virtual bool updateModule()
    {
        cv::Scalar colorGT = cv::Scalar(0, 0, 0.5);
        // plot ground-truth skeleton
        if (initTimer)
            drawSkeleton(poseGT, fullImg, colorGT);

        // plot tracked joint
        cv::Scalar colorTR = cv::Scalar(0.8, 0, 0);
        for(int i=0; i<N; i++)
        {
            while (!pose2img[i].empty())
            {
                int x = pose2img[i].front().u;
                int y = pose2img[i].front().v;
                cv::Point pt(x, y);
                cv::drawMarker(fullImg, pt, colorTR, 0, 6);
                pose2img[i].pop_front();
            }
        }
        drawSkeleton(pose, fullImg, colorTR);
        cv::Scalar colorRoi = cv::Scalar(0, 0.5, 0);
        if(roi) drawRoi(fullImg, colorRoi, 1);

        // double the resolution to add text
        cv::Mat aux;
        fullImg.copyTo(aux);
        cv::resize(fullImg, aux, cv::Size(dimX * 2, dimY * 2), 0, 0, cv::INTER_CUBIC);
        // Add text
        cv::putText(aux,
                    "Detection",
                    cv::Point(dimX * 1.6, dimY * 1.8), // Coordinates
                    cv::FONT_HERSHEY_SIMPLEX,          // Font
                    0.8,                               // Scale
                    cv::Scalar(0.0, 0.0, 0.8),         // BGR Color
                    1,                                 // Line Thickness
                    cv::LINE_AA);                      // Anti-alias
        cv::putText(aux,
                    "Tracking",
                    cv::Point(dimX * 1.6, dimY * 1.9), // Coordinates
                    cv::FONT_HERSHEY_SIMPLEX,          // Font
                    0.8,                               // Scale
                    cv::Scalar(0.8, 0.0, 0.0),         // BGR Color
                    1,                                 // Line Thickness
                    cv::LINE_AA);                      // Anti-alias
        std::string strF = std::to_string(int(avgF));
        std::string strdetF = std::to_string(int(detF));
        if (showF)
            cv::putText(aux,
                        "Freq = " + strF + "Hz" + " - detF = " + strdetF + "Hz",
                        cv::Point(dimX * 0.05, dimY * 1.9), // Coordinates
                        cv::FONT_HERSHEY_SIMPLEX,           // Font
                        0.8,                                // Scale
                        cv::Scalar(0.8, 0.8, 0.8),          // BGR Color
                        1,                                  // Line Thickness
                        cv::LINE_AA);                       // Anti-alias
            
        cv::putText(aux,
                    "HPE-core EDPR",
                    cv::Point(dimX * 0.5, dimY * 0.2), // Coordinates
                    cv::FONT_HERSHEY_SIMPLEX,          // Font
                    1.2,                               // Scale
                    cv::Scalar(0.8, 0.8, 0.8),         // BGR Color
                    1,                                 // Line Thickness
                    cv::LINE_AA);                      // Anti-alias

        if (plotCv) // output image using opencv
        {
            cv::imshow("HPE OUTPUT", aux);
            if(pastSurf)
            {
                cv::imshow("TOS", S.getSurface());
                cv::imshow("SAE", SAE_vis);
            } 
            cv::waitKey(1);
        }

        // output image using yarp
        aux *= 255;
        cv::Mat img_out;
        aux.convertTo(img_out, CV_8UC3);
        image_port.prepare().copy(yarp::cv::fromCvMat<PixelRgb>(img_out));
        image_port.write();
        fullImg = cv::Vec3f(0.4, 0.4, 0.4);

        return Thread::isRunning();
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
        double t0 = Time::now(), t1=t0, t2 = t0, t2prev = t0, t3[N] = {t0}, t4 = t0;
        // t0 = initial time
        // t1 = global timer
        // t2 = timer used for operation frequency calculation
        // t3 = timer used for fusion
        // t4 =  timer used for detection frequency downsampling
        long int mes = 0; // amount of porcessing cycles used to average operation frequency
        double freq = 0; // operation frequenc
        bool firstDet = false; // first detection took place
        
        while (!Thread::isStopping())
        {
            t1 = Time::now() - t0;
            // read detections
            // int N = input_sklt.getPendingReads();
            bot_sklt = input_sklt.read(false);
            input_sklt.getEnvelope(skltStamp);
            
            skltTs = skltStamp.getTime();

            ptJoint.x = int(poseGT[jointNum].u);
            ptJoint.y = int(poseGT[jointNum].v);
                
            if(bot_sklt && 1/(t1 - t4) <= detF) // there is a detection
            {
                Value &coords = (*bot_sklt).get(1);
                Bottle *sklt_lst = coords.asList();
                // build skeleton from reading
                skeleton13 builtPose = buildSklt(*sklt_lst);
                pose = tracker.resetPose(builtPose);   // reset pose
                poseGT = pose;
                // set roi for just all 13 joints
                for(int i=0; i<N; i++)
                {
                    int x = builtPose[i].u;
                    int y = builtPose[i].v;
                    qROI[i].setROI(x - roiWidth / 2, x + roiWidth / 2, y - roiHeight / 2, y + roiHeight / 2);
                }
                // output detections to file
                aux_out << t1 << " ";
                for (auto &t : builtPose)
                    aux_out << t.u << " " << t.v << " ";
                aux_out << std::endl;
                firstDet = true;
                t4 = Time::now() - t0;
            }

            // read events
            int np = input_port.queryunprocessed();
            if(np && initTimer)
            {
                t2 = Time::now() - t0;
                freq += 1/(t2-t2prev);
                mes++;
                avgF = freq/mes; // average freq
                if(mes>1000)
                {
                    mes= 0;
                    freq=0;
                }
                t2prev = t2;
                // avgF = 1/(t2-tprev); // instant freq
                // yInfo() << "\033c" << avgF;
            }
            for(int i=0; i<N; i++)
                nevs[i] = 0;
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
                    evsFullImg.push_back(qi); // save events to visualize in sync thread
                    for(int j=0; j<N; j++)
                    {
                        if(qi.x >= qROI[j].roi[0] && qi.x<qROI[j].roi[1] && qi.y >= qROI[j].roi[2] && qi.y<qROI[j].roi[3]) // if event qi falls inside roi, save int qROI
                        {
                            qROI[j].add(qi);
                            nevs[j]++;
                        }
                    }
                }
            }
            
        
            // Add events to output image
            if (initTimer)
                evsToImage(evsFullImg);

            // Process data for tracking
            if(pose.size() && firstDet) // a pose has been detected before
            {
                dpose = tracker.resetVel();
                for(int i=0; i<N; i++)
                {
                    if (nevs[i] && qROI[i].q.size() && !bot_sklt)// && qROI.q.front().stamp * vtsHelper::tsscaler > skltTs) // there are events to process
                    {
                        // separate events into deques to avoid using event-driven in hpe-core
                        std::deque<joint> evs;
                        std::deque<double> evsTs;
                        getEventsUV(qROI[i].q, evs, evsTs, vtsHelper::tsscaler); // get events u,v coords
                        
                        if(method == 1) // Velocity estimation Method 1: time diff on adjacent events 
                        {
                            qROI[i].setSize(int((qROI[i].roi[1] - qROI[i].roi[0]) * (qROI[i].roi[3] - qROI[i].roi[2])*1.5));
                            if(nevs[i] > 20)
                            {
                                dpose = tracker.method1(evs, evsTs, i, nevs[i], vels);  // get veocities from delta ts
                                double dt = (qROI[i].q.front().stamp - qROI[i].q.back().stamp) * vtsHelper::tsscaler;
                                tracker.fusion(&pose, dpose, dt);
                            }
                            else
                            {
                                dpose = tracker.resetVel();
                                double dt = (qROI[i].q.front().stamp - qROI[i].q.back().stamp) * vtsHelper::tsscaler;
                                tracker.fusion(&pose, dpose, dt);
                            }
                        }
                        else if(method == 2) // Velocity estimation method 2: neighbor events
                        {
                            int halfRoi = int((qROI[i].roi[1] - qROI[i].roi[0]) * (qROI[i].roi[3] - qROI[i].roi[2])*0.5);
                            if(pastSurf) qROI[i].setSize(nevs[i]);
                            else qROI[i].setSize(halfRoi);
                            // yInfo() << nevs;
                            // tracker.estimateFire(evs, evsTs, evsPol, jointNum, nevs, pose, dpose, Te, SAE);
                            // double err = tracker.getError(evs, evsTs, evsPol, jointNum, nevs, pose, dpose, Te, SAE);
                            // dpose = tracker.setVel(jointNum, dpose, pose[jointNum].u, pose[jointNum].v, err);
                            if(pastSurf) dpose[i] = tracker.estimateVelocity(evs, evsTs, nevs[i], vels, S.getSurface(), SAE);
                            else dpose[i] = tracker.estimateVelocity(evs, evsTs, nevs[i], vels);
                            double dt = t1 - t3[i];
                            tracker.fusion(&pose[i], dpose[i], dt);
                            t3[i] = t1;
                        }

                        // // write integrated pose output to file
                        // output_writer << t1 << " ";
                        // for (auto &t : pose)
                        //     output_writer << t.u << " " << t.v << " ";
                        // output_writer << std::endl;

                        // update roi
                        int x = pose[i].u;
                        int y = pose[i].v;
                        qROI[i].setROI(x - roiWidth / 2, x + roiWidth / 2, y - roiHeight / 2, y + roiHeight / 2);
                        pose2img[i].push_back(pose[i]);

                        // // output velocities estimations to file
                        // if(avgV) // true = write averaged vel - false = write event by event vel
                        // {
                        //     vel_out << t1 << " " << dpose[jointNum].u << " " << dpose[jointNum].v << std::endl;
                        // }
                        // else
                        // {
                        //     while(!vels.empty())
                        //     {
                        //         joint V = vels.front();
                        //         vel_out << t1 << " " << V.u << " " << V.v << std::endl;
                        //         vels.pop_front();
                        //     }
                        // }
                    }
                    // else if (bot_sklt) // there weren't events to process but a detection occured
                    // {
                    //     // write detected output to file
                    //     output_writer << t1 << " ";
                    //     for (auto &t : pose)
                    //         output_writer << t.u << " " << t.v << " ";
                    //     output_writer << std::endl;
                    // }
                }
            }
            if(pastSurf && initTimer)
            {
                for (auto &qi : *q)
                {
                    // S.EROSupdate(qi.x, qi.y);
                    S.TOSupdate(qi.x, qi.y);
                    SAE.at<float>(qi.y, qi.x) = float(qi.stamp*vtsHelper::tsscaler);
                }
                cv::normalize(SAE,SAE_vis, 0, 1, cv::NORM_MINMAX);
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
