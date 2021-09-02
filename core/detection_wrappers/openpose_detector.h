
#pragma once

#include "utility.h"
#include <opencv2/opencv.hpp>
#include <openpose/headers.hpp>

namespace hpecore {

class OpenPoseDetector {

    op::Wrapper detector{op::ThreadManagerMode::Asynchronous};
    int poseJointsNum;

  public:
    bool init(std::string poseModel, std::string poseMode);
    skeleton detect(cv::Mat &image);
    void stop();
};

}