#include "utility.h"
#include "motion.h"

namespace hpecore{

const std::vector< std::vector<cv::Point> > pwtripletvelocity::is = {{{0,0},{1,1}},
                                                                    {{0,2},{1,2}}, 
                                                                    {{0,4},{1,3}}, 
                                                                    {{2,4},{2,3}}, 
                                                                    {{4,4},{3,3}}, 
                                                                    {{4,2},{3,2}}, 
                                                                    {{4,0},{3,1}}, 
                                                                    {{2,0},{2,1}}};

const std::vector<jDot> pwtripletvelocity::vs ={{1,  1}, 
                                                {1,  0}, 
                                                {1, -1}, 
                                                {0, -1}, 
                                                {-1,-1}, 
                                                {-1, 0}, 
                                                {-1, 1}, 
                                                {0,  1}};

pwtripletvelocity::wjv pwtripletvelocity::point_velocity(const cv::Mat &local_sae)
{

    wjv wvel = {{0,0}, 0};

    for(size_t i = 0; i < is.size(); i++) 
    {
        const double &t0 = local_sae.at<double>(2, 2);
        const double &t1 = local_sae.at<double>(is[i][1]);
        const double &t2 = local_sae.at<double>(is[i][0]);
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


jDot pwtripletvelocity::area_velocity(const cv::Mat &area_sae)
{
    wjv wvel = {{0,0}, 0};
    auto s = area_sae.size();
    for(int y = 2; y < s.height-2; y++) {
        for(int x = 2; x < s.width-2; x++) {
            const double &ts = area_sae.at<double>(y, x);
            if(ts > prev_update_ts)
                wvel += point_velocity(area_sae({x-2,y-2,5,5}));
        }
    }
    if(wvel.c > 0)
        return {wvel.v.u/wvel.c, wvel.v.v/wvel.c};
    else
        return {0, 0};
}

std::vector<jDot> pwtripletvelocity::multi_area_velocity(const cv::Mat &full_sae, double ts, std::vector<joint> js, int radius)
{
    std::vector<jDot> output;

    for(auto j : js) {
        cv::Rect joint_region = cv::Rect(j.u - radius, j.v - radius, radius*2+1, radius*2+1) & cv::Rect({0, 0}, full_sae.size());
        if(joint_region.width < 5 || joint_region.height < 5)
            output.push_back({0, 0});
        else 
            output.push_back(area_velocity(full_sae(joint_region)));
    }
    prev_update_ts = ts;
    return output;
}

skeleton13 pwtripletvelocity::multi_area_velocity(const cv::Mat &full_sae, double ts, skeleton13 js, int radius)
{
    skeleton13 output; 
    int i = 0;
    for(auto j : js) {
        cv::Rect joint_region = cv::Rect(j.u - radius, j.v - radius, radius*2+1, radius*2+1) & cv::Rect({0, 0}, full_sae.size());
        if(joint_region.width < 5 || joint_region.height < 5)
            output[i++] = {0, 0};
        else 
            output[i++] = area_velocity(full_sae(joint_region));
    }
    prev_update_ts = ts;
    return output;
}



}