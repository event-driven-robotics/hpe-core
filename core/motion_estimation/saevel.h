#include <opencv2/opencv.hpp>
#include "utility.h"

namespace hpecore {
class saevel
{
public:

cv::Mat sae;
cv::Mat layer;
cv::Mat mask;
cv::Point vel;

void initialise(int width, int height)
{
    sae = cv::Mat(height, width, CV_64F); sae.setTo(0.0);
    layer = cv::Mat(height, width, CV_64F); layer.setTo(0.0);
    mask = cv::Mat(height, width, CV_8U); mask.setTo(0);
}

template<typename T>
void update(const T &begin, const T &end)
{
    for(auto v = begin; v != end; v++) {
        if(mask.at<char>(v->y, v->x) > 0 || v.timestamp() - sae.at<double>(v->y, v->x) < 0.1)
            continue; 
        layer.at<double>(v->y, v->x) = v.timestamp();
        mask.at<char>(v->y, v->x) = 255;
    }
}

template <typename T>
void update(const T &packet, double ts)
{
    update<T::iterator>(packet.begin(), packet.end());
}

cv::Mat makeImage(cv::Mat xsae, cv::Mat xlayer, cv::Mat xmask, double t)
{
    double minv, maxv;
    cv::minMaxLoc(xsae, &minv, &maxv);
    cv::Mat temp_8, temp_64, temp_bgr;
    temp_64 = xsae - (maxv - t);
    temp_64.convertTo(temp_8, CV_8U, 255.0/t);
    std::vector<cv::Mat> imgvec = {temp_8, xmask, cv::Mat::zeros(xmask.size(), CV_8U)};
    cv::merge(imgvec, temp_bgr);

    cv::resize(temp_bgr, temp_bgr, mask.size());

    return temp_bgr;
}

cv::Mat makeImage(double t = 1.0)
{
    return makeImage(sae, layer, mask, t);
}

cv::Mat makeImage(int x, int y, int r, double t = 1.0)
{
    static cv::Rect fullroi = cv::Rect(cv::Point(0, 0), mask.size());
    cv::Rect roi = cv::Rect(x-r, y-r, r*2, r*2);
    roi &= fullroi;

    cv::Mat img = makeImage(sae(roi), layer(roi), mask(roi), t);
    //cv::Point c(img.cols/2, img.rows/2);

    //cv::line(img, c, c+vel*20, cv::Vec3b(255, 255, 255));

    return img;

}

jDot convolution(int x, int y, int r, cv::Mat &img)
{
    static cv::Rect fullroi = cv::Rect(cv::Point(0, 0), mask.size());
    static int search = 9;
    
    cv::Rect roi = cv::Rect(x-r, y-r, r*2, r*2);
    roi &= fullroi;

    if(cv::countNonZero(layer(roi)) < 5)
        return {0, 0};

    cv::Mat heat_map;

    cv::filter2D(layer(roi), heat_map, CV_64F, sae(roi), cv::Point(-1, -1), 0.0, cv::BORDER_ISOLATED);

    cv::Rect z(heat_map.cols/2-search, heat_map.rows/2-search, search*2+1, search*2+1);

    heat_map = heat_map(z);
    cv::GaussianBlur(heat_map, heat_map, cv::Size(7, 7), -1);
    //heat_map.at<double>(search, search) = 0;
    double minv, maxv;
    cv::minMaxLoc(heat_map, &minv, &maxv, (cv::Point *)0, &vel);
    vel -= cv::Point(search, search);

    //calculate the dt for this shift
    cv::Mat shifted;
    cv::Mat M = (cv::Mat_<float>(2, 3) << 1, 0, vel.x,
                                          0, 1, vel.y);
    cv::warpAffine(sae(roi), shifted, M, roi.size());
    cv::Mat dta = layer(roi) - shifted;
    cv::threshold(dta, dta, 0.0, 0.0, cv::THRESH_TOZERO);
    float dtm = cv::mean(dta, mask(roi))[0];  


    heat_map.copyTo(img);
    cv::normalize(img, img, -10.0, 1.0, cv::NORM_MINMAX);
    cv::resize(img, img, mask.size(), 0.0, 0.0, cv::INTER_NEAREST);
    
    
    return {vel.x/dtm, vel.y/dtm};

}

void conlude_update()
{
    layer.copyTo(sae, mask);
    layer.setTo(0.0);
    mask.setTo(0);
}

};

}
