//TODO license

#pragma once

#include <tuple>
#include <vector>
#include <iostream>

namespace hpecore {
using skeleton = std::vector<std::tuple<double, double>>;

inline void print_skeleton(const skeleton s) 
{
    for(auto &t : s)
        std::cout << std::get<0>(t) << " " << std::get<1>(t) << std::endl;
}

}
