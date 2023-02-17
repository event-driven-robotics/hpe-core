/* BSD 3-Clause License

Copyright (c) 2021, Event Driven Perception for Robotics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */
#pragma once

#include <opencv2/core.hpp>
#include <string>

#include "utility.h"

namespace hpecore {

// pure integration
class stateEstimator {
   protected:
    bool pose_initialised{false};
    skeleton13 state{0};
    skeleton13_vel velocity{0};
    std::deque<hpecore::jDot> error[13];
    double prev_ts{0.0};

   public:
    virtual bool initialise(std::vector<double> parameters = {});
    virtual void updateFromVelocity(jointName name, jDot velocity, double ts);
    virtual void updateFromPosition(jointName name, joint position, double ts);
    virtual void updateFromVelocity(skeleton13 velocity, double ts);
    virtual void updateFromPosition(skeleton13 position, double ts);
    virtual void set(skeleton13 pose, double ts);
    virtual bool poseIsInitialised();
    virtual skeleton13 query();
    virtual joint query(jointName name);
    skeleton13_vel queryVelocity();
    void setVelocity(skeleton13_vel vel);
    std::deque<hpecore::jDot>* queryError();
};

// constant position model
class kfEstimator : public stateEstimator {
   private:
    std::array<cv::KalmanFilter, 13> kf_array;
    double procU{0.0}, measUV{0.0}, measUD{0.0};

    void setTimePeriod(double dt);

   public:
    bool initialise(std::vector<double> parameters) override;
    void updateFromVelocity(skeleton13 velocity, double ts) override;
    void updateFromVelocity(jointName name, jDot velocity, double ts) override;
    void updateFromPosition(jointName name, joint position, double ts) override;
    void updateFromPosition(skeleton13 position, double ts) override;
    void set(skeleton13 pose, double ts) override;
};

// constant velocity model
class constVelKalman : public stateEstimator {
   private:
    std::array<cv::KalmanFilter, 13> kf_array;
    double procU{0.0}, measU{0.0}, procUd{0.0}, measUd{0.0};
    cv::Mat measurement_velocity, measurement_position;

    void setTimePeriod(double dt);

   public:
    bool initialise(std::vector<double> parameters) override;
    void updateFromVelocity(skeleton13 velocity, double ts) override;
    void updateFromVelocity(jointName name, jDot velocity, double ts) override;
    void updateFromPosition(jointName name, joint position, double ts) override;
    void updateFromPosition(skeleton13 position, double ts) override;
    void set(skeleton13 pose, double ts) override;

};

// latency compensation constant position model
class singleJointLatComp {
   private:
    cv::KalmanFilter kf;
    joint vel_accum;
    std::deque<joint> history_v;
    std::deque<double> history_ts;
    double measU;
    double procU;
    double prev_pts;
    double prev_vts;

   public:
    void initialise(double procU, double measU) {
        this->procU = procU;
        this->measU = measU;

        // init(state size, measurement size)
        kf.init(2, 2);
        kf.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
        kf.measurementMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
        kf.processNoiseCov = (cv::Mat_<float>(2, 2) << procU, 0, 0, procU);
        kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) << measU, 0, 0, measU);
    }

    void set(joint position, double ts) {
        kf.statePost.at<float>(0) = position.u;
        kf.statePost.at<float>(1) = position.v;
        prev_pts = ts;
        prev_vts = ts;
        vel_accum = {0, 0};
    }

    void updateFromVelocity(jDot velocity, double ts) {
        history_v.push_back(velocity * (ts - prev_vts));
        history_ts.push_back(ts);
        vel_accum = vel_accum + history_v.back();
        prev_vts = ts;
    }

    void updateFromPosition(joint position, double ts, bool use_comp = true)
    {
        size_t i = 0; //position in history
        //if not using latency compensation we want to integrate before doing
        //the position update.
        if(!use_comp)
        {
            kf.statePost.at<float>(0) += vel_accum.u;
            kf.statePost.at<float>(1) += vel_accum.v;
        }
        //if we use latency compensation we want to integrate only up to the
        //point the detection was made
        else
        {
            for(;i < history_ts.size(); i++)
            {
                if(history_ts[i] > ts) break;
                kf.statePost.at<float>(0) += history_v[i].u;
                kf.statePost.at<float>(1) += history_v[i].v;
            }
        }

        // perform an (asynchronous) update of the position from the previous
        // position estimated
        double dt = ts - prev_pts;
        kf.processNoiseCov.at<float>(0, 0) = procU * dt;
        kf.processNoiseCov.at<float>(1, 1) = procU * dt;
        kf.predict();
        if(position.u > 0 || position.v > 0)
            kf.correct((cv::Mat_<float>(2, 1) << position.u, position.v));

        // add the remaining current period velocity accumulation to the state
        if (use_comp)
        {
            for(;i < history_ts.size(); i++)
            {
                kf.statePost.at<float>(0) += history_v[i].u;
                kf.statePost.at<float>(1) += history_v[i].v;
            }
        }
        vel_accum = {0.0, 0.0};
        history_v.clear();
        history_ts.clear();
        prev_pts = ts;
        
    }

    joint query() {
        return {kf.statePost.at<float>(0) + vel_accum.u,
                kf.statePost.at<float>(1) + vel_accum.v};
    }
};

class multiJointLatComp : public stateEstimator {
   private:
    std::array<singleJointLatComp, 13> kf_array;
    bool use_lc{false};

   public:
    bool initialise(std::vector<double> parameters) override {
        if (parameters.size() < 4)
            return false;
        for (auto &j : kf_array)
            j.initialise(parameters[0], parameters[1]);
        if(parameters[3] > 0.0) use_lc = true;
        return true;
    }

    void updateFromVelocity(jointName name, jDot velocity, double ts) override {
        kf_array[name].updateFromVelocity(velocity, ts);
        state[name] = kf_array[name].query();
    }

    void updateFromVelocity(skeleton13 velocity, double ts) override {
        for (auto name : jointNames)
            updateFromVelocity(name, velocity[name], ts);
    }

    void updateFromPosition(jointName name, joint position, double ts) override {
        kf_array[name].updateFromPosition(position, ts, use_lc);
        state[name] = kf_array[name].query();
    }

    void updateFromPosition(skeleton13 position, double ts) override {
        for (auto name : jointNames)
            updateFromPosition(name, position[name], ts);
    }

    void set(skeleton13 position, double ts) override {
        pose_initialised = true;
        state = position;
        for (auto name : jointNames)
            kf_array[name].set(position[name], ts);
    }
};

}  // namespace hpecore
