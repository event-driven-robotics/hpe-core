#include <opencv2/opencv.hpp>
#include <deque>
#include "utility.h"

namespace hpecore {

template <typename T> 
inline void createCountImage(std::deque<T> &input, cv::Mat &output)
{
    output.setTo(0);
    for (auto &v : input)
        output.at<uchar>(v.x, v.y)++;
}

void varianceNormalisation(cv::Mat &input);

}