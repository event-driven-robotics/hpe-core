#include "representations.h"

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



const cv::Mat &surface::getSurface() 
{
    return surf;
}

void surface::init(int width, int height, int kernel_size, double parameter) 
{
    if (kernel_size % 2 == 0)
        kernel_size++;
    this->kernel_size = kernel_size;  //kernel_size should be odd
    this->half_kernel = kernel_size / 2;
    this->parameter = parameter;

    roi_full = cv::Rect(0, 0, width, height);
    roi_raw = cv::Rect(0, 0, kernel_size, kernel_size);
    surf = cv::Mat(height, width, CV_8U, cv::Scalar(0));
}

bool surface::setRoiAndRegion(int x, int y) 
{
    //set the roi around the event location
    roi_raw.x = x - half_kernel;
    roi_raw.y = y - half_kernel;
    //get only the roi in the limits of the surface
    roi_valid = roi_raw & roi_full;
    //set the region
    region = surf(roi_valid);
    //if roi_raw != roi_valid this was an event next to a border
    return !(roi_raw == roi_valid);
}

bool surface::TOSupdate(const int vx, const int vy) 
{
    //parameter default = 2
    static const int threshold = 255 - kernel_size * parameter;

    bool border = setRoiAndRegion(vx, vy);

    unsigned char &c = region.at<unsigned char>(half_kernel, half_kernel);
    for (auto x = 0; x < region.cols; x++) {
        for (auto y = 0; y < region.rows; y++) {
            unsigned char &p = region.at<unsigned char>(y, x);
            p < threshold ? p = 0 : p--;
        }
    }
    c = 255;

    return border;
}

bool surface::SITSupdate(const int vx, const int vy) 
{
    static const int maximum_value = kernel_size * kernel_size;

    bool border = setRoiAndRegion(vx, vy);

    unsigned char &c = region.at<unsigned char>(half_kernel, half_kernel);
    for (auto x = 0; x < region.cols; x++) {
        for (auto y = 0; y < region.rows; y++) {
            unsigned char &p = region.at<unsigned char>(y, x);
            if (p > c) p--;
        }
    }
    c = maximum_value;

    return border;
}

bool surface::EROSupdate(const int vx, const int vy) 
{
    //parameter default = 0.3
    static double odecay = pow(parameter, 1.0 / kernel_size);

    bool border = setRoiAndRegion(vx, vy);

    unsigned char &c = region.at<unsigned char>(half_kernel, half_kernel);
    for (auto x = 0; x < region.cols; x++) {
        for (auto y = 0; y < region.rows; y++) {
            unsigned char &p = region.at<unsigned char>(y, x);
            p *= odecay;
        }
    }
    c = 255;

    return border;
}
}