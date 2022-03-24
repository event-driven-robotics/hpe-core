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

#include <deque>
#include <math.h>
#include "utility.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace hpecore
{
  class jointMotionEstimator
  {

    public:
       
      skeleton13 resetPose(skeleton13 detection);
      skeleton13 resetVel();
      skeleton13 estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int nevs, std::deque<joint>& vels);
      skeleton13 estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int nevs, std::deque<joint>& vels, cv::Mat eros, cv::Mat timeSurf);
      void fusion(skeleton13 *pose, skeleton13 dpose, double dt);

      // Funtiones using joints instead of skeleton13
      joint estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs, int nevs, std::deque<joint>& vels);
      joint estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs, int nevs, std::deque<joint>& vels, cv::Mat eros, cv::Mat timeSurf);
      void fusion(joint *pose, joint dpose, double dt);

      // Functions no longer being used
      // method one for velocity estimation
      skeleton13 method1(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int nevs, std::deque<joint>& vels);
      // Estimated time of fire functions
      void estimateFire(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, skeleton13 pose, skeleton13 dpose, double **Te, cv::Mat matTe);
      double getError(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, skeleton13 pose, skeleton13 dpose, double **Te, cv::Mat matTe);
      joint centerMass(std::deque<joint> evs,  int jointName, int nevs, skeleton13 pose);
      skeleton13 setVel(int jointName, skeleton13 dpose, double dx, double dy, double err);
      
  };
}