#pragma once

#include <deque>
#include <math.h>
#include "utility.h"

namespace hpecore
{
  class jointMotionEstimator
  {

    public:
       
      sklt resetPose(sklt detection);
      sklt estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs);
      void fusion(sklt *pose, sklt dpose, double dt);

      template <typename T>
      inline void getEventsUV(std::deque<T> &input, std::deque<joint> &output, std::deque<double> &ts, double scaler) 
      {
        for (auto &q : input)
        {
          // joint j;
          // j.u = q.x;
          // j.v = q.y;
          joint j = {q.x, q.y};
          output.push_back(j);
          ts.push_back(q.stamp*scaler);
        }
      }
   
     
  };
}