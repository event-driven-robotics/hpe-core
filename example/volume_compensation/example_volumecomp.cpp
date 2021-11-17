#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <event-driven/all.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <hpe-core/volumes.h>
#include <mutex>
#include <fstream>
#include "visualisation.h"
#include <event-driven/vIPT.h>


// #define f 274.573
// #define x0 139.228
// #define y0 123.092

#define f 500
#define x0 152
#define y0 120


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

    pixelShifter ps;
    cv::Mat blank_iso;

public:

    flowVisualiser() {}
    void initialise(int height, int width, double window_size) {

        masker = cv::Mat(height, width, CV_8U);
        debug_image = cv::Mat(height, width, CV_8UC3);
        arrow_image = cv::Mat(height, width, CV_8UC3);
        compensated_image = cv::Mat(height, width, CV_32FC3);
        reset();

        ps = drawISOBase(height, width, window_size, blank_iso);

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

    void drawISO(const std::deque<AE> &volume, std::string name)
    {
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::Mat base_copy;
        blank_iso.copyTo(base_copy);
        for (auto &c : volume) {
            int x = c.x;
            int y = c.y;
            double z = ev::vtsHelper::deltaS(volume.back().stamp, c.stamp);
            ps.pttr(x, y, z);
            if (x >= 0 && y >= 0 && x < base_copy.cols && y < base_copy.rows)
                base_copy.at<cv::Vec3b>(y, x) = cv::Vec3b(120, 120, 120);
        }
        for (auto &c : volume) {
            int x = c.x;
            int y = c.y;
            double z = ev::vtsHelper::deltaS(volume.back().stamp, c.stamp);
            ps.pttr(x, y, z);
            base_copy.at<cv::Vec3b>(c.y, x) = cv::Vec3b(255, 255, 255);
        }
        cv::imshow(name, base_copy);
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

    imuAdvHelper imu_helper;
    double period{0.1};
    double window_size{0.1};
    bool rot_only{false};

    resolution res{304, 240};

    deque<AE> window;
    flowVisualiser fv;
    std::mutex m;

    std::fstream vel_writer;

public:

    volumeCompensator(){};

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        if(rf.check("help") || rf.check("h")) {
            yInfo() << "Event Volumer and Compensator";
            yInfo() << "--height <int>: image height";
            yInfo() << "--width <int> : image width";
            yInfo() << "--period <float>: image output rate";
            yInfo() << "--rot_only <bool>: use 3DoF rotation only";
            yInfo() << "--window <float>: period of data to compensate";
            yInfo() << "--vel_dump <string>: path to file to dump velocities";
            yInfo() << "--imu <string>: path to imu calibration file";
            yInfo() << "--imu_beta <double>: AHRS beta parameter";
            return false;
        }

        //set the module name used to name ports
        setName((rf.check("name", Value("/volumer")).asString()).c_str());

        res.height = rf.check("height", Value(240)).asInt();
        res.width  = rf.check("width",  Value(304)).asInt();
        period = rf.check("period", Value(0.05)).asDouble();
        rot_only = rf.check("rot_only", Value(false)).asBool();
        window_size = rf.check("window", Value(0.1)).asDouble();
        std::string vel_dump_path = "";
        if(rf.check("vel_dump"))
            vel_dump_path = rf.find("vel_dump").asString();

        if(!vel_dump_path.empty()) {
            vel_writer.open(vel_dump_path, std::ios_base::out|std::ios_base::trunc);
            if(!vel_writer.is_open()) {
                yError() << "Could not open supplied writer path" << vel_dump_path;
                return false;
            }
            vel_writer << std::fixed << std::setprecision(6);
        }

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

        fv.initialise(res.height, res.width, window_size);

        if(rf.check("imu"))
            imu_helper.loadIMUCalibrationFiles(rf.find("imu").asString());
        
        if(rf.check("imu_beta"))
            imu_helper.setBeta(rf.find("imu_beta").asDouble());

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
        vel_writer.close();
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
            const vector<IMUevent> *q_imu = imu_port.read(yarpstamp_imu);
            if (!q_imu || Thread::isStopping()) return false;
            imu_helper.addIMUPacket(*q_imu);
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
        hpecore::camera_velocity cam_vel = imu_helper.extractVelocity();
        double temp3 = cam_vel[0]; cam_vel[0] = cam_vel[1]; cam_vel[1] = -temp3;
        //cam_vel[0] = 0.0; cam_vel[1] = 0.0; cam_vel[2] = 0.0;
        hpecore::camera_params   cam_par = {f, x0, y0};
        fv.reset();
        deque<AE> warped_volume;
        m.lock();
        for (auto &v : window) {
            hpecore::point_flow pf = hpecore::estimateVisualFlow({(double)v.x, (double)v.y, 0.0}, cam_vel, cam_par);
            auto we = hpecore::spatiotemporalWarp(v, pf, vtsHelper::deltaS(window.back().stamp, v.stamp));
            fv.addPixel(v.x, v.y, we.x, we.y, pf.udot, pf.vdot, period);
            warped_volume.push_back(we);
        }
        fv.drawISO(window, "Original Events");
        fv.drawISO(warped_volume, "Warped Events");
        std::array<double, 4> qd = imu_helper.extractHeading();
        //for(auto i : qd) std::cout << i; std::cout << std::endl;
        //std::array<float, 4> qf = qd;
        cv::Mat temp = ev::drawRefAxis(qd);
        cv::imshow("temp", temp);
        cv::waitKey(1);
        m.unlock();

        fv.show();

        yarp::sig::Vector &yarp_vel = scope_port.prepare();
        yarp_vel.resize(6);
        std::array<double, 6> cam_acc = imu_helper.extractAcceleration();
        static std::array<double, 6> mean_acc = {0, 0, 0, 0, 0, 0};
        static std::array<double, 6> mean_vel = {0, 0, 0, 0, 0, 0};
        static std::array<double, 6> my_vel = {0, 0, 0, 0, 0, 0};
        std::array<double, 3> gv = imu_helper.getGravityVector();
        static double tic = yarp::os::Time::now();
        double dt = yarp::os::Time::now() - tic; tic += dt;
        for(auto i = 0; i < 3; i++) {
            if(dt > 1.0) dt = 1.0;
            mean_acc[i] = mean_acc[i]*(1.0 - dt) + cam_acc[i] * dt;
            yarp_vel[i] = mean_acc[i];
            yarp_vel[i] = gv[i];
            //yarp_vel[i] = cam_acc[i] - mean_acc[i];
            yarp_vel[i+3] = cam_acc[i];// - gv[i];
            //if(fabs(cam_acc[i] - mean_acc[i]) > 0.1)
            //double temp_vel = my_vel[i] + (cam_acc[i] - mean_acc[i]) * dt * 0.5;
            //mean_vel[i] = mean_vel[i]*(1.0-dt) + temp_vel*dt;
            my_vel[i] += (cam_acc[i] - gv[i]) * dt;
            //my_vel[i] *= 0.99;
            
            //yarp_vel[i] = my_vel[i];
        }
        scope_port.write();

        if (vel_writer.is_open()) {
            vel_writer << yarpstamp_imu.getTime() << " ";
            for (auto &i : cam_vel)
                vel_writer << i << " ";
            vel_writer << std::endl;
        }

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


