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

#include <opencv2/opencv.hpp>
#include <deque>
#include <array>
#include "utility.h"

namespace hpecore {

// Image creation

template <typename T>
inline void createCountImage(std::deque<T> &input, cv::Mat &output) 
{
    output.setTo(0);
    for (auto &v : input)
        output.at<uchar>(v.x, v.y)++;
}

void varianceNormalisation(cv::Mat &input);

// Event warping
using camera_velocity = std::array<double, 6>;

typedef struct camera_params
{
    float f;
    float cx;
    float cy;
} camera_params;

typedef struct pixel_3d
{
    double x{0};
    double y{0};
    double d{0};
} pixel_3d;

point_flow estimateVisualFlow(pixel_3d px3, const camera_velocity &vcam, const camera_params &pcam);



template <typename T>
T spatiotemporalWarp(T original, point_flow flow, double deltat) 
{
    T output;
    output.stamp = original.stamp + deltat;
    output.y = original.y + flow.vdot * deltat + 0.5;
    output.x = original.x + flow.udot * deltat + 0.5;
    return output;
}

class surface 
{
private:
    int kernel_size{0};
    int half_kernel{0};
    cv::Rect roi_full, roi_raw, roi_valid;
    double parameter{0};
    cv::Mat surf;
    cv::Mat region;

    bool setRoiAndRegion(int x, int y);

public:
   
    const cv::Mat& getSurface();
    void init(int width, int height, int kernel_size, double parameter = 0.0);
    bool TOSupdate(const int vx, const int vy);
    bool SITSupdate(const int vx, const int vy);
    bool EROSupdate(const int vx, const int vy);
};

    

    

}  // namespace hpecore