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

class stateEstimator {
   protected:
    bool pose_initialised{false};
    skeleton13 state{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

   public:
    virtual bool initialise(std::vector<double> parameters = {}) {
        return true;
    }
    virtual void updateFromVelocity(jointName name, jDot velocity, double dt) {
        state[name] += (velocity * dt);
    }

    virtual void updateFromPosition(jointName name, joint position, double dt) {
        state[name] = position;
    }

    virtual void updateFromVelocity(skeleton13 velocity, double dt) {
    }

    virtual void updateFromPosition(skeleton13 position, double dt) {
        state = position;
    }

    virtual void set(skeleton13 pose) {
        state = pose;
        pose_initialised = true;
    }

    bool poseIsInitialised() {
        return pose_initialised;
    }

    skeleton13 query() {
        return state;
    }

    joint query(jointName name) {
        return state[name];
    }
};

class kfEstimator : public stateEstimator {
   private:
    std::array<cv::KalmanFilter, 13> kf_array;
    double procU{0.0}, measU{0.0};

    void setTimePeriod(double dt) {
        for (auto &kf : kf_array) {
            kf.processNoiseCov.at<float>(0, 0) = procU * dt;
            kf.processNoiseCov.at<float>(1, 1) = procU * dt;
        }
    }

   public:
    bool initialise(std::vector<double> parameters) override {
        if (parameters.size() != 2)
            return false;
        procU = parameters[0];
        measU = parameters[1];

        // init(state size, measurement size)
        for (auto &kf : kf_array) {
            kf.init(2, 2);
            kf.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 0,
                                   0, 1);

            kf.measurementMatrix = (cv::Mat_<float>(2, 2) << 1, 0,
                                    0, 1);

            kf.processNoiseCov = (cv::Mat_<float>(2, 2) << procU, 0,
                                  0, procU);
            kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) << measU, 0,
                                      0, measU);
        }
        return true;
    }

    void updateFromVelocity(jointName name, jDot velocity, double dt) override {
        auto &kf = kf_array[name];
        joint j_new = state[name] + velocity * dt;
        setTimePeriod(dt);
        kf.predict();
        kf.correct((cv::Mat_<float>(2, 1) << j_new.u, j_new.v));
        state[name] = {kf.statePost.at<float>(0), kf.statePost.at<float>(1)};
    }

    void updateFromPosition(jointName name, joint position, double dt) override {
        auto &kf = kf_array[name];
        setTimePeriod(dt);
        kf.predict();
        kf.correct((cv::Mat_<float>(2, 1) << position.u, position.v));
        state[name] = {kf.statePost.at<float>(0), kf.statePost.at<float>(1)};
    }

    void updateFromPosition(skeleton13 position, double dt) override {
       for(auto &name : jointNames)
          updateFromPosition(name, position[name], dt);
    }

    void set(skeleton13 pose) override {
        state = pose;
        for (jointName name : jointNames) {
            auto &kf = kf_array[name];
            kf.statePost.at<float>(0) = state[name].u;
            kf.statePost.at<float>(1) = state[name].v;
        }
        pose_initialised = true;
    }

    bool poseIsInitialised() {
        return pose_initialised;
    }

    skeleton13 query() {
        return state;
    }

    joint query(jointName name) {
        return state[name];
    }
};

}  // namespace hpecore
