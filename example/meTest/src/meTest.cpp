#include <yarp/os/all.h>
#include <event-driven/all.h>
#include "meTest.h"

using namespace ev;
using namespace yarp::os;

roiq qROI;
std::mutex m;

roiq::roiq()
{
    roi.resize(4);
    n = 1000;
    roi[0] = 0; roi[1] = 1000;
    roi[2] = 0; roi[3] = 1000;
    use_TW = false;
}

void roiq::setSize(unsigned int value)
{
    //if TW n is in clock-ticks
    //otherwise n is in # events.
    n = value;
    while(q.size() > n)
        q.pop_back();
}

void roiq::setROI(int xl, int xh, int yl, int yh)
{
    roi[0] = xl; roi[1] = xh;
    roi[2] = yl; roi[3] = yh;
}

int roiq::add(const ev::AE &v)
{

    if(v.x < roi[0] || v.x > roi[1] || v.y < roi[2] || v.y > roi[3])
        return 0;
    q.push_front(v);
    return 1;
}

bool plotIMage(int waitTime)
{
//    cv::Mat debug_image;
    cv::Mat test = cv::Mat::zeros(cv::Size(346, 240), CV_32FC1);

//    m.lock();
//    vpf.debug_particle.appearance.copyTo(debug_image);
    test += 2;
    test *= 0.2;
//    vpf.debug_particle.score = 0;
    for(auto v = qROI.q.rbegin(); v < qROI.q.rend(); v++) {

//        float &value = vpf.debug_particle.appearance.at<float>(v->y, v->x);

        int value = v->polarity;
//        std::cout << v->x << "\t" << v->y << "\t" << value << std::endl;
        if(v->x<346 && v->y<240)
        {
            if(value > 0)
            {
                test.at<float>(v->y, v->x) = 0.0f;
            }
            if(value <= 0)
            {
                test.at<float>(v->y, v->x) = 1.0f;
            }
        }

//        vpf.debug_particle.score += value*0.5;
    }

    m.unlock();

    cv::Mat frame;
    test.convertTo(frame, CV_8U, 255);
    cv::imshow("Frame", frame);
    cv::waitKey(waitTime);
    return true;

}

class exampleModule : public RFModule, public Thread {

private:

    vReadPort< vector<AE> > input_port;
    vWritePort output_port;

    bool example_flag;
    double example_parameter;

public:

    exampleModule() {}

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        //set the module name used to name ports
        setName((rf.check("name", Value("/file")).asString()).c_str());

        /* initialize yarp network */
        yarp::os::Network yarp;
        if(!yarp.checkNetwork(2.0)) {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }

        //open io ports
        if(!input_port.open(getName() + "/AE:i")) {
            yError() << "Could not open input port";
            return false;
        }
        output_port.setWriteType(AE::tag);
        if(!output_port.open(getName() + "/AE:o")) {
            yError() << "Could not open input port";
            return false;
        }

        yarp.connect("/file/ch3dvs:o", getName("/AE:i"), "fast_tcp");

        //read flags and parameters
        example_flag = rf.check("example_flag") &&
                rf.check("example_flag", Value(true)).asBool();
        double default_value = 0.1;
        example_parameter = rf.check("example_parameter",
                                     Value(default_value)).asDouble();

        //do any other set-up required here
        cv::namedWindow("Frame", cv::WINDOW_NORMAL);

        //start the asynchronous and synchronous threads
        return Thread::start();
    }

    virtual double getPeriod()
    {
        return 1.0; //period of synchrnous thread
    }

    bool interruptModule()
    {
        //if the module is asked to stop ask the asynchrnous thread to stop
        return Thread::stop();
    }

    void onStop()
    {
        //when the asynchrnous thread is asked to stop, close ports and do
        //other clean up
        input_port.close();
    }

    //synchronous thread
    virtual bool updateModule()
    {

        //add any synchronous operations here, visualisation, debug out prints


        return Thread::isRunning();
    }

    //asynchronous thread run forever
    void run()
    {
        Stamp yarpstamp;
        deque<AE> out_queue;
        unsigned int i = 0;
        int targetproc = 500;
        int desired_nw = 200*4;

        yarp::os::Stamp ystamp;

//        while(true) {
//
//            const vector<AE> * q = input_port.read(yarpstamp);
//            if(!q || Thread::isStopping()) return;
//
//            //do asynchronous processing here
//            for(auto &qi : *q) {
//
//                //here you could try modifying the data of the event before
//                //pushing to the output q
//
//                //the position of the event (qi.x, qi.y)
//                //the polarity of the event (qi.polarity)
//                //the timstamp (qi.stamp)
//                //or remove some events based on a condition?
//                //if(qi.x < 100) only takes the left 1/3 of events
//
//                out_queue.push_back(qi);
//                std::cout << out_queue.front().stamp * vtsHelper::tsscaler<< "\t" << qi.stamp * vtsHelper::tsscaler<< "\t" << qi.x << "\t" << qi.y << std::endl;
//            }
//
//            //after processing the packet output the results
//            //(only if there is something to output
//            if(out_queue.size()) {
//                output_port.write(out_queue, yarpstamp);
//                out_queue.clear();
//            }
//        }



        const vector<AE> *q = input_port.read(ystamp);
        if(!q || Thread::isStopping()) return;
        qROI.setROI(0, 346, 0, 260);

        while(true)
        {
            std::cout << "loop\n";
            int addEvents = 0;
            while(addEvents < targetproc)
            {
                if(i >= q->size())
                {
                    i = 0;
                    m.unlock();
                    q = input_port.read(ystamp);
                    m.lock();
                    if(!q || Thread::isStopping()) {
                        m.unlock();
                        return;
                    }
                }
                addEvents += qROI.add((*q)[i]);
                i++;
            }
            plotIMage(1);
            qROI.setSize(0);
//            m.unlock();
        }
    }

};

int main(int argc, char * argv[])
{
    /* initialize yarp network */
    yarp::os::Network yarp;
    if(!yarp.checkNetwork(2)) {
        std::cout << "Could not connect to YARP" << std::endl;
        return false;
    }

    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setVerbose( false );
    rf.setDefaultContext( "eventdriven" );
    rf.setDefaultConfigFile( "sample_module.ini" );
    rf.configure( argc, argv );

    /* create the module */
    exampleModule instance;
    return instance.runModule(rf);
}
