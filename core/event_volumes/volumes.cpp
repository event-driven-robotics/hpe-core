#include "volumes.h"

using namespace cv;
using namespace std;

void hpecore::varianceNormalisation(cv::Mat &input) 
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
