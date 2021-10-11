#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <event-driven/all.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <hpe-core/volumes.h>
#include <mutex>
//#include <dirent.h>
//#include <sstream>
//#include <numeric>
//#include <iterator>

#define f 274.573
#define x0 139.228
//#define x0 152.228
#define y0 123.092

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;
using std::vector;
using std::string;

class flowVisualiser {
private:
    cv::Mat masker;
    cv::Mat debug_image;
    cv::Mat arrow_image;
    cv::Mat compensated_image;
    const static int arrow_space{20};

public:

    flowVisualiser() {}
    void initialise(int height, int width) {

        masker = cv::Mat(height, width, CV_8U);
        debug_image = cv::Mat(height, width, CV_8UC3);
        arrow_image = cv::Mat(height, width, CV_8UC3);
        compensated_image = cv::Mat(height, width, CV_32FC3);
        reset();

    }

    void reset() {
        masker.setTo(1);
        debug_image.setTo(128);
        arrow_image.setTo(0);
        compensated_image.setTo(cv::Scalar(0, 0, 0));
        
    }

    void addPixel(int u, int v, int uhat, int vhat, double udot, double vdot, double period) 
    {
        //update the original uncompensated image
        debug_image.at<cv::Vec3b>(v, u) = cv::Vec3b(255, 255, 255);

        //update the compensated image with HSV calculated colours
        if (vhat >= 0 && vhat < compensated_image.rows && uhat >= 0 && uhat < compensated_image.cols) {
            double magnitude = 255.0;
            magnitude *= sqrt(udot * udot + vdot * vdot) / 200.0;
            if (magnitude > 255.0) magnitude = 255.0;
            if (magnitude < 1.0) magnitude = 1.0;
            double angle = atan2(vdot, udot);
            angle = 180.0 * (angle + M_PI) / (2.0 * M_PI);
            compensated_image.at<cv::Vec3f>(vhat, uhat) = cv::Vec3f(angle, 255.0f, magnitude);
        }

        //if an arrow hasn't been drawn nearby, draw one
        if (masker.at<unsigned char>(v, u)) {
            cv::line(arrow_image, cv::Point(u, v),
                     cv::Point(u + udot * period, v + vdot * period),
                     CV_RGB(1, 1, 1));

            //and update the masker to say an arrow is already drawn
            //in this region
            int xl = std::max(u - arrow_space, 0);
            int xh = std::min(u + arrow_space, masker.cols);
            int yl = std::max(v - arrow_space, 0);
            int yh = std::min(v + arrow_space, masker.rows);
            masker(cv::Rect(cv::Point(xl, yl), cv::Point(xh, yh))).setTo(0);
        }
    }

    void show()
    {

        //build hsv image
        cv::Mat bgr, hsv8;
        compensated_image.convertTo(hsv8, CV_8UC3);
        cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
        bgr.copyTo(debug_image, bgr);
        arrow_image.copyTo(debug_image, arrow_image);

        cv::namedWindow("Compensated", cv::WINDOW_NORMAL);
        cv::imshow("Compensated", debug_image);
        cv::waitKey(3);

    }

};

class volumeCompensator : public RFModule,
                          public Thread {
   private:

    vReadPort< vector<AE> > vis_port;
    vReadPort< vector<IMUevent>  > imu_port;
    BufferedPort< Vector > scope_port;

    vector<double> imu_state;
    imuHelper imu_helper;
    double period;
    bool dof6;
    double window_size;

    resolution res;

    deque<AE> window;
    flowVisualiser fv;
    std::mutex m;

public:

    volumeCompensator()
    {
        imu_state.resize(10, 0.0);
    }

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        if(rf.check("help") || rf.check("h")) {
            yInfo() << "Event Volumer and Compensator";
            yInfo() << "--height <int>: image height";
            yInfo() << "--width <int> : image width";
            yInfo() << "--period <float>: image output rate";
            yInfo() << "--dof6  <bool>: use all 6 DoF";
            yInfo() << "--window <float>: period of data to compensate";
            return false;
        }

        //set the module name used to name ports
        setName((rf.check("name", Value("/volumer")).asString()).c_str());

        res.height = rf.check("height", Value(240)).asInt();
        res.width  = rf.check("width",  Value(304)).asInt();
        period = rf.check("period", Value(0.1)).asDouble();
        dof6 = rf.check("dof6", Value(false)).asBool();
        window_size = rf.check("window", Value(0.1)).asDouble();

        Network yarp;
        if(!yarp.checkNetwork(2.0)) {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }

        //open io ports
        if(!vis_port.open(getName("/AE:i"))) {
            yError() << "Could not open input port";
            return false;
        }

        if(!imu_port.open(getName("/IMU:i"))) {
            yError() << "Could not open input port";
            return false;
        }

        if(!scope_port.open(getName() + "/scope:o")) {
            yError() << "Could not open scope out port";
            return false;
        }

        yarp.connect("/vPreProcess/imu_samples:o",
                     getName("/IMU:i"),
                     "fast_tcp");

        yarp.connect("/vPreProcess/right:o",
                     getName("/AE:i"),
                     "fast_tcp");

        fv.initialise(res.height, res.width);

        //start the asynchronous and synchronous threads
        return Thread::start();
    }

    virtual double getPeriod()
    {
        return period; //period of synchrnous thread
    }

    bool interruptModule() //RFMODULE
    {
        //if the module is asked to stop ask the asynchrnous thread to stop
        return Thread::stop();
    }

    void onStop() //THREAD
    {
        //when the asynchrnous thread is asked to stop, close ports and do
        //other clean up
        vis_port.close();
        imu_port.close();
        scope_port.close();
    }

    //synchronous thread
    virtual bool updateModule()
    {

        //get all available updates to the IMU, then push them to the 
        //new IMU helper that does orientation estimation and velocity
        //estimation. extract the velocity vector
        static Stamp yarpstamp_imu;
        auto n_packets = imu_port.queryunprocessed();

        //update the current value of the imu sensor values
        for (auto i = 0; i < n_packets; i++) {
            // const vector<IMUevent> * q_imu = imu_port.read(yarpstamp);

            const vector<IMUevent> *q_imu = imu_port.read(yarpstamp_imu);
            if (!q_imu || Thread::isStopping()) return false;

            for (auto &v : *q_imu) {
                imu_state[v.sensor] = imu_helper.convertToSI(v.value, v.sensor);
            }
        }
        if(window.empty()) return true;

        //update the madgwick filter
        // double dtime = vtsHelper::deltaMS(timestamp, p_timestamp);
        // orientation_filter.setPeriod(dtime);
        // p_timestamp = timestamp;

        // orientation_filter.AHRSupdateIMU(imu_state[imuHelper::GYR_X],
        //                                  imu_state[imuHelper::GYR_Y],
        //                                  imu_state[imuHelper::GYR_Z],
        //                                  imu_state[imuHelper::ACC_X],
        //                                  imu_state[imuHelper::ACC_Y],
        //                                  imu_state[imuHelper::ACC_Z]);

        // //perform compensation
        // auto gp = orientation_filter.gravity_compensate(
        //     imu_state[imuHelper::ACC_X],
        //     imu_state[imuHelper::ACC_Y],
        //     imu_state[imuHelper::ACC_Z]);
        // vel_estimate.integrate(gp, dtime);
        
        //form the velocity vector correctly
        // static vector<double> vel_imu(6, 0.0);
        // vel_imu[0] = vel_estimate.getLinearVelocity()[0];
        // vel_imu[1] = vel_estimate.getLinearVelocity()[1];
        // vel_imu[2] = vel_estimate.getLinearVelocity()[2];
        // vel_imu[3] = imu_state[imuHelper::GYR_X];
        // vel_imu[4] = imu_state[imuHelper::GYR_Y];
        // vel_imu[5] = imu_state[imuHelper::GYR_Z];

        

        double period = vtsHelper::deltaS(window.back().stamp, window.front().stamp);
        yInfo() << period;
        hpecore::camera_velocity cam_vel = {0, 0, 0, imu_state[imuHelper::GYR_X], imu_state[imuHelper::GYR_Y], imu_state[imuHelper::GYR_Z]};
        hpecore::camera_params   cam_par = {f, x0, y0};
        fv.reset();
        m.lock();
        for (auto &v : window) {
            hpecore::point_flow pf = hpecore::estimateVisualFlow({v.x, v.y, 0}, cam_vel, cam_par);
            auto we = hpecore::spatiotemporalWarp(v, pf, vtsHelper::deltaS(window.back().stamp, v.stamp));
            fv.addPixel(v.x, v.y, we.x, we.y, pf.udot, pf.vdot, period);
        }
        m.unlock();

        fv.show();

        yarp::sig::Vector &yarp_vel = scope_port.prepare();
        yarp_vel.resize(6);
        for(auto i = 0; i < 6; i++)
            yarp_vel[i] = cam_vel[i];
        scope_port.write();
        

        return Thread::isRunning();
    }

    //asynchronous thread run forever
    void run()
    {
        Stamp yarpstamp;
        while(true) {
            const vector<AE> * q_flow = vis_port.read(yarpstamp);
            if(!q_flow || Thread::isStopping()) 
                return;
            
            m.lock();
            for(auto &v : *q_flow)
                if(v.x >= 0 && v.y >= 0 && v.x < res.width && v.y < res.height)
                    window.push_back(v);

            while(ev::vtsHelper::deltaS(window.back().stamp, window.front().stamp) > window_size)
                window.pop_front();
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
    volumeCompensator instance;
    return instance.runModule(rf);
}
