#pragma once

#include <deque>
#include <math.h>
#include "utility.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace hpecore
{
  class jointMotionEstimator
  {

    public:
       
      sklt resetPose(sklt detection);
      sklt estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int k, sklt dpose);
      void fusion(sklt *pose, sklt dpose, double dt);

      template <typename T>
      inline void getEventsUV(std::deque<T> &input, std::deque<joint> &output, std::deque<double> &ts, double scaler, std::deque<int> &pol) 
      {
        for (auto &q : input)
        {
          joint j = {q.x, q.y};
          output.push_back(j);
          ts.push_back(q.stamp*scaler);
          pol.push_back(q.polarity);
        }
      }

      cv::Mat projNew(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt dpose, double tnow);
      cv::Mat projPrev(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt dpose, double tnow);
      sklt comparteProjs(int jointName, sklt dpose, cv::Mat newEvs, cv::Mat oldEvs);
  };
}