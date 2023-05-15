/* BSD 3-Clause License

Copyright (c) 2023, Event Driven Perception for Robotics
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

#include "utility.h"
#include <opencv2/opencv.hpp>
#include <array>
namespace hpecore {

class skeletonCFT {
private:

      typedef struct intrinsic {
         int w;
         int h;
         int u0;
         int v0;
         double fu;
         double fv;
         double k1;
         double k2;
      } intrinsic;

      intrinsic cam1;
      intrinsic cam2;

      cv::Mat T;

public:

   void setCam1Parameters(std::array<int, 4> res, std::array<double, 4> dis)
   {
      cam1.w = res[0]; cam1.h = res[1]; cam1.u0 = res[2]; cam1.v0 = res[3];
      cam1.fu = dis[0]; cam1.fv = dis[1]; cam1.k1 = dis[2]; cam1.k2 = res[3];
   }

   void setCam2Parameters(std::array<int, 4> res, std::array<double, 4> dis)
   {
      cam2.w = res[0]; cam2.h = res[1]; cam2.u0 = res[2]; cam2.v0 = res[3];
      cam2.fu = dis[0]; cam2.fv = dis[1]; cam2.k1 = dis[2]; cam2.k2 = res[3];
   }

   void setExtrinsicTransform(std::array<double, 16> E) 
   {
      T = cv::Mat(4, 4, CV_32F, E.data());
   }

   joint cft(joint in, double depth)
   {
      //undistort point
      joint und1;
      double r2 = (in.u - cam1.u0)*(in.u - cam1.u0) + (in.v - cam1.v0)*(in.v - cam1.v0);
      double r4 = r2*r2;
      double den = 1 + cam1.k1 * r2 + cam1.k2 * r4;
      und1.u = cam1.u0 + (in.u - cam1.u0) / den;
      und1.v = cam1.v0 + (in.v - cam1.v0) / den;

      //project to 3D
      cv::Mat X1 = (cv::Mat_<double>(4,1) << depth * und1.u / cam1.fu, depth * und1.v / cam1.fv, depth, 1.0);

      //transform point to second camera reference
      cv::Mat X2 = T * X1;

      //project to 2D
      joint und2;
      und2.u = cam2.fu * X2.at<double>(0, 0) / X2.at<double>(0, 2) + cam2.u0;
      und2.v = cam2.fv * X2.at<double>(0, 1) / X2.at<double>(0, 2) + cam2.v0;

      //distort point
      double ru2 = (und2.u - cam2.u0)*(und2.u - cam2.u0) + (und2.v - cam2.v0)*(und2.v - cam2.v0);
      double F = (1 - sqrt(1 - 4*cam2.k1*ru2)) / (2 * cam2.k1 * ru2);
      joint out;
      out.u = cam2.u0 + (und2.u - cam2.u0) * F;
      out.v = cam2.v0 + (und2.v - cam2.v0) * F;

      return out;
   }

   skeleton13 cft(skeleton13 in, double depth)
   {
      skeleton13 out = {0};
      int i = 0;
      for(auto &j : in)
         out[i++] = cft(j);
      return out;
   }




};

}

