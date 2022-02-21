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
      sklt resetVel();
      sklt estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int k, sklt dpose, std::deque<joint>& vels);
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

      void estimateFire(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt pose, sklt dpose, double **Te, cv::Mat matTe);
      double getError(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt pose, sklt dpose, double **Te, cv::Mat matTe);
      joint centerMass(std::deque<joint> evs,  int jointName, int nevs, sklt pose);
      sklt setVel(int jointName, sklt dpose, double dx, double dy, double err);
      sklt nearestEvent(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int nevs, sklt dpose, std::deque<joint>& vels);
  };
}