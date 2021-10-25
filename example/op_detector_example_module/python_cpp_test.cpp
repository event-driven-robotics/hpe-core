
#include "e2vid.h"
#include <yarp/os/all.h>
#include <iostream>
#include <vector>


int main(int argc, char * argv[])
{
    /* prepare and configure the resource finder */
//    yarp::os::ResourceFinder rf;
//    rf.setVerbose( false );
//    rf.configure( argc, argv );

    int n = 10000;
    double t1 = 0, t2 = 1, nextTerm = 0;
    std::vector<double> vect;

    std::cout << "Fibonacci Series: ";

    for (int i = 1; i <= n; ++i) {
        // Prints the first two terms.
        if(i == 1) {
            std::cout << t1 << ", ";
            continue;
        }
        if(i == 2) {
            std::cout << t2 << ", ";
            continue;
        }
        nextTerm = t1 + t2;
        t1 = t2;
        t2 = nextTerm;
        vect.push_back(t1);
        vect.push_back(t2);

        std::cout << nextTerm << ", ";
    }

    std::cout << std::endl << "----------------------------------vector size:  " << vect.size();

    E2Vid e2vid;
    e2vid.init(260, 346, 7500, 0.35);




    std::string model_root = "/usr/local/e2vid/python/rpg_e2vid/pretrained";
    PyObject* args = Py_BuildValue("(iiids)",
                                   5,
                                   5,
                                   5,
                                   0.5,
                                   model_root.c_str()
                                   );
    if(args == NULL)
    {
        std::cout << "could not build python value" << std::endl;
    }
    std::cout << "build python value finished" << std::endl;


    e2vid.init_model(260, 346, 7500, 0.35);

    if (!yarp::os::Network::checkNetwork(2.0)) {
        std::cout << "Could not connect to YARP" << std::endl;
    }
    std::cout << "network checked" << std::endl;

    return 0;
}
