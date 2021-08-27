
#include <openpose/headers.hpp>

#include "../utility/utility.h"


class OpenPoseDetector {

    op::Wrapper detector;
    int poseJointsNum;

  public:
    OpenPoseDetector(std::string poseModel, poseMode);
    skeleton detect(cv::Mat image);
}