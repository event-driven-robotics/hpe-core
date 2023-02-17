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

#include "utility.h"
#include <opencv2/opencv.hpp>
#include <openpose/headers.hpp>

namespace hpecore {

class OpenPoseDetector {

    op::Wrapper detector{op::ThreadManagerMode::Asynchronous};
    int poseJointsNum;

  public:
    bool init(std::string poseModel, std::string poseMode, std::string size = "256");
    skeleton13 detect(cv::Mat &input);
    void stop();
};

class openposethread
{
private:
    std::thread th;
    hpecore::OpenPoseDetector detop;
    hpecore::stampedPose pose{0.0, -1.0, 0.0};
    cv::Mat image;

    bool stop{false};
    bool data_ready{true};
    std::mutex m;

    void run();

public:
    bool init(std::string model_path, std::string model_name, 
              std::string model_size = "256");
    void close();
    bool update(cv::Mat next_image, double image_timestamp, 
                hpecore::stampedPose &previous_result);
};

}