#include "volumes.h"

using namespace cv;
using namespace std;

namespace hpecore {

void varianceNormalisation(cv::Mat &input) 
{
    constexpr double threshold = 0.1 / 255;

    //get the average value of any pixel
    int count_unique = cv::countNonZero(input);
    double mean_pval = cv::sum(input)[0] / count_unique;

    //calculate the variance of the (non-zero) pixels
    double var = 0;
    for (auto x = 0; x < input.cols; x++) {
        for (auto y = 0; y < input.rows; y++) {
            auto &p = input.at<uchar>(x, y);
            if (p > 0) {
                double e = p - mean_pval;
                var += e * e;
            }
        }
    }
    var /= count_unique;
    double sigma = sqrt(var);
    if (sigma < threshold) sigma = threshold;

    //scale each pixel by the scale factor
    double scale_factor = 255.0 / (3.0 * sigma);
    for (auto x = 0; x < input.cols; x++) {
        for (auto y = 0; y < input.rows; y++) {
            auto &p = input.at<uchar>(x, y);
            if (p > 0) {
                double v = p * scale_factor;
                if (v > 255.0) v = 255.0;
                if (v < 0.0) v = 0.0;
                p = (uchar)v;
            }
        }
    }
}

point_flow estimateVisualFlow(pixel_3d px3, const camera_velocity &vcam, const camera_params &pcam) 
{
    
    double f_inv = 1.0 / pcam.f;
    double x = px3.x - pcam.cx;
    double y = px3.y - pcam.cy;
    point_flow flow;
    flow.udot = (x * y * f_inv)*vcam[3] + -(pcam.f + pow(x, 2) * f_inv)*vcam[4] +  y*vcam[5];
    flow.vdot = (pcam.f + pow(y, 2) * f_inv)*vcam[3] + (-x * y * f_inv)*vcam[4] + -x*vcam[5];

    if (px3.d) {
        flow.udot = flow.udot + (-pcam.f / px3.d) * vcam[0] + x / px3.d * vcam[2];
        flow.vdot = flow.vdot + (-pcam.f / px3.d) * vcam[1] + y / px3.d * vcam[2];
    }
    return flow;
}

}
