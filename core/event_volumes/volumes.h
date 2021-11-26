#include <opencv2/opencv.hpp>
#include <deque>
#include <array>
#include "utility.h"

namespace hpecore {

// Image creation

template <typename T>
inline void createCountImage(std::deque<T> &input, cv::Mat &output) 
{
    output.setTo(0);
    for (auto &v : input)
        output.at<uchar>(v.x, v.y)++;
}

void varianceNormalisation(cv::Mat &input);

// Event warping
using camera_velocity = std::array<double, 6>;

typedef struct camera_params
{
    float f;
    float cx;
    float cy;
} camera_params;

typedef struct pixel_3d
{
    double x{0};
    double y{0};
    double d{0};
} pixel_3d;

point_flow estimateVisualFlow(pixel_3d px3, const camera_velocity &vcam, const camera_params &pcam);



template <typename T>
T spatiotemporalWarp(T original, point_flow flow, double deltat) 
{
    T output;
    output.stamp = original.stamp + deltat;
    output.y = original.y + flow.vdot * deltat + 0.5;
    output.x = original.x + flow.udot * deltat + 0.5;
    return output;
}

}  // namespace hpecore