/* BSD 3-Clause License

Copyright (c) 2021, Event Driven Perception for Robotics
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
#pragma once

#include "utility.h"
#include <opencv2/opencv.hpp>

namespace hpecore 
{

class pwtripletvelocity
{
private:
    double tolerance{0.15};
    double prev_update_ts;
    static const std::vector< std::vector<cv::Point> > is;
    static const std::vector<jDot> vs;

    typedef struct wjv {
        jDot v;
        int  c;
        wjv& operator+=(const wjv& rhs) {
            this->c += rhs.c;
            this->v = {this->v.u + rhs.v.u, this->v.v + rhs.v.v};
            return *this;
        }
    } wjv;

    inline wjv point_velocity(const cv::Mat &local_sae)
    {

        wjv wvel = {{0,0}, 0};

        for(size_t i = 0; i < is.size(); i++) 
        {
            const double &t0 = local_sae.at<double>(2, 2);
            const double &t1 = local_sae.at<double>(is[i][1]);
            const double &t2 = local_sae.at<double>(is[i][2]);
            double dta = t0-t1;
            double dtb = t1-t2;
            bool valid = dta > 0 && dtb > 0 && t1 > 0 && t2 > 0;
            if(!valid) continue;
            double error = fabs(1 - dtb/dta);
            if(error > tolerance) continue;
            //valid triplet. calulate the velocity.
            double invt = 2.0 /  (dta + dtb);
            wvel.v.u += vs[i].u * invt;
            wvel.v.v += vs[i].v * invt;
            wvel.c++;
            //test rectification of the vector here. in theory slowest vector is most accurate.
        }
        return wvel;
    }

public:

    jDot area_velocity(const cv::Mat &area_sae);
    std::vector<jDot> multi_area_velocity(const cv::Mat &full_sae, double ts, std::vector<joint> js, int radius);

};



}