
#include "jointMotionEstimator.h"

using namespace hpecore;


sklt jointMotionEstimator::resetPose(sklt detection)
{
    return detection;
}

sklt jointMotionEstimator::estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs)
{
    sklt vel;
    for (size_t i = 0; i < vel.size(); i++)
    {
        vel[i].u = 0;
        vel[i].v = 0;
    }
    // calculate velocity for hand L joint
    double dtx = 0.0, dty = 0.0;
    for(std::size_t i = 0; i < evs.size()-1; i++)
    {
        // std::cout << evsTs[i] << " ";
        for(std::size_t j = i+1; j < evs.size(); j++)
        {
            if(evs[i].v == evs[j].v)// same y value
            {
                if(evs[i].u == evs[j].u+1)
                    dtx += (evsTs[i] - evsTs[j]);
                else if(evs[i].u == evs[j].u-1)
                    dtx -= (evsTs[i] - evsTs[j]);
            }
            if(evs[i].u == evs[j].u)// same x value
            {
                if(evs[i].v == evs[j].v+1)
                    dty += (evsTs[i] - evsTs[j]);
                else if(evs[i].v == evs[j].v-1)
                    dty -= (evsTs[i] - evsTs[j]);
            }
        }
    }
    vel[8].u = dtx * 10; 
    vel[8].v = dty * 10;
    // double norm = 1/(dtx*dtx + dty*dty);
    // vel[handL].u = dtx * norm;
    // vel[handL].v = dty * norm;
    // std::cout << std::endl;
    return vel;
}

void jointMotionEstimator::fusion(sklt *pose, sklt dpose, double dt)
{
    for (size_t i = 0; i < (*pose).size(); i++)
    {
        (*pose)[i].u = (*pose)[i].u + dpose[i].u * dt;
        (*pose)[i].v = (*pose)[i].v + dpose[i].v * dt;
    }
}