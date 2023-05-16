/* BSD 3-Clause License

Copyright (c) 2023, Event Driven Perception for Robotics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

#include <opencv2/opencv.hpp>
#include <hpe-core/reprojection.h>
#include <math.h>

int main(int argc, char * argv[])
{
   static const int w = 640;
   static const int h = 480;
   static const int w2 = 960;
   static const int h2 = 540;

   static constexpr double theta = 1 * M_PI / 180.0;


   
   hpecore::skeletonCFT mycft;

   

   mycft.setCam1Parameters({w, h, w/2, h/2}, {320, 320, -0.016, 0});
   mycft.setCam2Parameters({w2, h2, w2/2, h2/2}, {500, 500, -0.016, 0});
   // mycft.setExtrinsicTransform({cos(theta), -sin(theta), 0, 1, 
   //                                  sin(theta), cos(theta), 0, 0.2, 
   //                                  0, 0, 1, 0, 
   //                                  0, 0, 0, 1});

   mycft.setExtrinsicTransform({0.999946, 0.0046135, 0.00935247, 0.019063, -0.0047823, 0.999825, 0.0181072, 0.0977093, -0.00926729, -0.0181509, 0.999792, 0.114073, 0, 0, 0, 1});

   cv::Mat img  = cv::Mat::zeros(h, w, CV_8UC3);
   cv::Mat img2 = cv::Mat::zeros(h2, w2, CV_8UC3);
   //cv::Mat output_image = cv::Mat::zeros(w, h, CV_8U);

   hpecore::skeleton13 js = {0};
   js[0] = {308, 89}; js[1] = {279, 113}; js[2] = {323, 119};
   js[3] = {323, 119}; js[4] = {267, 144}; js[5] = {334, 170};
   js[6] = {328, 209}; js[7] = {292, 211}; js[8] = {308, 122};
   js[9] = {341, 213}; js[10] = {293, 301}; js[11] = {326, 300};
   js[12] = {282, 391};

   auto out = mycft.cft(js, 3);

   hpecore::drawSkeleton(img, js, {0, 0, 200});
   hpecore::drawSkeleton(img2, out, {200, 0, 200});



   cv::imshow("Transform", img);
   cv::imshow("Transform2", img2);
   cv::waitKey(10000);

   return 0;
}