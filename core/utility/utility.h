//TODO license

#pragma once

#include <tuple>
#include <vector>
#include <iostream>

namespace hpecore {

typedef struct pixel_event
{
    int p:1;
    int x:10;
    int _f1:1;
    int y:9;
    int _f:11;
    int ts:32;
} pixel_event;

using skeleton = std::vector<std::tuple<double, double>>;

inline void print_skeleton(const skeleton s) 
{
    for(auto &t : s)
        std::cout << std::get<0>(t) << " " << std::get<1>(t) << std::endl;
}

}