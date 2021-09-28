
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
    for (int i = 0; i < 1; i++)
    {
        
    }
}