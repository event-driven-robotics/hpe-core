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
#include "representations.h"

using namespace cv;
using namespace std;

namespace hpecore {

void varianceNormalisation(cv::Mat &input) 
{
    constexpr double threshold = 0.1 / 255;

    //get the average value of any pixel
    int count_unique = cv::countNonZero(input);
    double mean_pval = cv::sum(input)[0] / count_unique;

    //calculate the variance of the (non-zero) pixels
    double var = 0;
    for (auto x = 0; x < input.cols; x++) {
        for (auto y = 0; y < input.rows; y++) {
            auto &p = input.at<uchar>(x, y);
            if (p > 0) {
                double e = p - mean_pval;
                var += e * e;
            }
        }
    }
    var /= count_unique;
    double sigma = sqrt(var);
    if (sigma < threshold) sigma = threshold;

    //scale each pixel by the scale factor
    double scale_factor = 255.0 / (3.0 * sigma);
    for (auto x = 0; x < input.cols; x++) {
        for (auto y = 0; y < input.rows; y++) {
            auto &p = input.at<uchar>(x, y);
            if (p > 0) {
                double v = p * scale_factor;
                if (v > 255.0) v = 255.0;
                if (v < 0.0) v = 0.0;
                p = (uchar)v;
            }
        }
    }
}

point_flow estimateVisualFlow(pixel_3d px3, const camera_velocity &vcam, const camera_params &pcam) 
{
    
    double f_inv = 1.0 / pcam.f;
    double x = px3.x - pcam.cx;
    double y = px3.y - pcam.cy;
    point_flow flow;
    flow.udot = (x * y * f_inv)*vcam[3] + -(pcam.f + pow(x, 2) * f_inv)*vcam[4] +  y*vcam[5];
    flow.vdot = (pcam.f + pow(y, 2) * f_inv)*vcam[3] + (-x * y * f_inv)*vcam[4] + -x*vcam[5];

    if (px3.d) {
        flow.udot = flow.udot + (-pcam.f / px3.d) * vcam[0] + x / px3.d * vcam[2];
        flow.vdot = flow.vdot + (-pcam.f / px3.d) * vcam[1] + y / px3.d * vcam[2];
    }
    return flow;
}


cv::Mat surface::getSurface() 
{
    return surf(actual_region);
}

void surface::init(int width, int height, int kernel_size, double parameter) 
{
    if (kernel_size % 2 == 0)
        kernel_size++;
    this->kernel_size = kernel_size;  //kernel_size should be odd
    this->half_kernel = kernel_size / 2;
    this->parameter = parameter;

    surf = cv::Mat(height+half_kernel*2, width+half_kernel*2, CV_64F, cv::Scalar(0.0));
    actual_region = {half_kernel, half_kernel, width, height};
}

void surface::temporalDecay(double ts, double alpha) {
    surf *= cv::exp(alpha * (time_now - ts));
    time_now = ts;
}

void surface::spatialDecay(int k) 
{
    cv::GaussianBlur(surf, surf, cv::Size(k, k), 0);
}

}