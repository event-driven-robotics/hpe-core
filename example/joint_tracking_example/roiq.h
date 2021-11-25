#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <event-driven/all.h>

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;

class roiq
{
public:

    std::deque<ev::AE> q;
    unsigned int n;
    yarp::sig::Vector roi;
    bool use_TW;

    roiq();
    void setSize(unsigned int value);
    void setROI(int xl, int xh, int yl, int yh);
    int add(const ev::AE &v);

};