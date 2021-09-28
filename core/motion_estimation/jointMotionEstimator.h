#pragma once

#include "utility.h"

namespace hpecore
{
  class jointMotionEstimator
  {

    public:
       
      skeleton resetPose(skeleton detection);
      skeleton estimateVelocity();
      void fusion(skeleton *pose, skeleton dpose);
     
  };
}