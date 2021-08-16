#include <iostream>
#include "../detection_wrappers/detection.h"

#include <yarp/os/all.h>
#include <event-driven/all.h>
using namespace ev;
using namespace yarp::os;

using namespace hpecore;
using namespace cv;
using std::tuple;

using namespace std;

int main(int argc, char * argv[])
{
    std::cout << "Hello world!" << std::endl;
    hpecore::skeleton joint;
    std::cout << get<0>(joint) << " + " << std::tuple_size<decltype(joint)>::value << std::endl;
    return 0;
}
