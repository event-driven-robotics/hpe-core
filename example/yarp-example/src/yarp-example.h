#include <yarp/os/all.h>
#include <yarp/sig/Vector.h>
#include <event-driven/all.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <sstream>
#include <deque>
#include <SuperimposeMesh/SICAD.h>

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

bool plotIMage(int waitTime);
