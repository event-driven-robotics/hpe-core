//TODO license

#pragma once

#include <tuple>
#include <vector>
#include <array>
#include <iostream>
#include <string>

namespace hpecore {
using skeleton = std::vector<std::tuple<double, double>>;

typedef struct {
    float u;
    float v;
} joint ;

typedef std::array<joint, 13> sklt;
enum skltJoint {head, shoulderR, shoulderL, elbowR, elbowL,
             hipL, hipR, handR, handL, kneeR, kneeL, footR, footL};

inline skltJoint str2enum(const std::string& str)
{
    if(str == "head") return head;
    else if(str == "shoulderR") return shoulderR;
    else if(str == "shoulderL") return shoulderL;
    else if(str == "elbowR") return elbowR;
    else if(str == "elbowL") return elbowL;
    else if(str == "hipL") return hipL;
    else if(str == "hipR") return hipR;
    else if(str == "handR") return handR;
    else if(str == "handL") return handL;
    else if(str == "kneeR") return kneeR;
    else if(str == "kneeL") return kneeL;
    else if(str == "footR") return footR;
    else if(str == "footL") return footL;
}

inline void print_skeleton(const skeleton s) 
{
    for(auto &t : s)
        std::cout << std::get<0>(t) << " " << std::get<1>(t) << std::endl;
}

inline void print_sklt(const sklt s) 
{
    for(auto& j : s)
        std::cout << j.u << " " << j.v << std::endl;
}

}