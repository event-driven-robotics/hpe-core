//TODO license

#pragma once

#include <tuple>
#include <vector>
#include <array>
#include <iostream>

namespace hpecore {
using skeleton = std::vector<std::tuple<double, double>>;

typedef struct {
    float u;
    float v;
} joint ;

typedef std::array<joint, 13> sklt;
enum motion {head, shoulderR, shoulderL, elbowR, elbowL,
             hipL, hipR, handR, handL, kneeR, kneeL, footR, footL};

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