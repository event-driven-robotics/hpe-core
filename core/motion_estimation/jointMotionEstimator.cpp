
#include "jointMotionEstimator.h"

using namespace hpecore;


// reset function that doesn't do anything
skeleton jointMotionEstimator::resetPose(skeleton detection)
{
    return detection;
}

skeleton jointMotionEstimator::estimateVelocity()
{
    skeleton vel;
    for (size_t i = 0; i < 13; i++)
    {
        vel.emplace_back(std::make_tuple(0, 0));
    }
    return vel;
}

void jointMotionEstimator::fusion(skeleton *pose, skeleton dpose)
{
    for (size_t i = 0; i < (*pose).size(); i++)
    {
        std::get<0>((*pose).at(i)) += std::get<0>((dpose).at(i));
        std::get<1>((*pose).at(i)) += std::get<0>((dpose).at(i));
    }
}