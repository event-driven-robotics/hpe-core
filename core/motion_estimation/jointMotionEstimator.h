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
      sklt estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int nevs, std::deque<joint>& vels);
      void fusion(sklt *pose, sklt dpose, double dt);

      // Functions no longer being used
      // method one for velocity estimation
      sklt method1(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int nevs, std::deque<joint>& vels);
      // Estimated time of fire functions
      void estimateFire(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt pose, sklt dpose, double **Te, cv::Mat matTe);
      double getError(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt pose, sklt dpose, double **Te, cv::Mat matTe);
      joint centerMass(std::deque<joint> evs,  int jointName, int nevs, sklt pose);
      sklt setVel(int jointName, sklt dpose, double dx, double dy, double err);
      
  };
}