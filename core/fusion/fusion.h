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
    skeleton13 state{0};
    skeleton13_vel velocity{0};
    std::deque<hpecore::jDot> error[13];

   public:
    virtual bool initialise(std::vector<double> parameters = {});
    virtual void updateFromVelocity(jointName name, jDot velocity, double dt);
    virtual void updateFromPosition(jointName name, joint position, double dt);
    virtual void updateFromVelocity(skeleton13 velocity, double dt);
    virtual void updateFromPosition(skeleton13 position, double dt);
    virtual void set(skeleton13 pose);
    bool poseIsInitialised();
    skeleton13 query();
    joint query(jointName name);
    skeleton13_vel queryVelocity();
    void setVelocity(skeleton13_vel vel);
    std::deque<hpecore::jDot>* queryError();
};

class kfEstimator : public stateEstimator {
   private:
    std::array<cv::KalmanFilter, 13> kf_array;
    double procU{0.0}, measU{0.0};

    void setTimePeriod(double dt);

   public:
    bool initialise(std::vector<double> parameters) override;
    void updateFromVelocity(jointName name, jDot velocity, double dt) override;
    void updateFromPosition(jointName name, joint position, double dt) override;
    void updateFromPosition(skeleton13 position, double dt) override;
    void set(skeleton13 pose) override;
    bool poseIsInitialised();
    skeleton13 query();
    joint query(jointName name);
};

}  // namespace hpecore
