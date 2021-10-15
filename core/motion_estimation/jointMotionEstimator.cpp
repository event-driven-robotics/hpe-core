
#include "jointMotionEstimator.h"

using namespace hpecore;


sklt jointMotionEstimator::resetPose(sklt detection)
{
    return detection;
}

sklt jointMotionEstimator::estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs, int jointName)
{
    /*
    sklt vel;
    for (size_t i = 0; i < vel.size(); i++)
    {
        vel[i].u = 0;
        vel[i].v = 0;
    }
    // calculate velocity for hand L joint
    double dtx = 0.0, dty = 0.0, k=1e3;
    long int nx = 0, ny = 0;
    for(std::size_t i = 0; i < evs.size(); i++)
    {
        // std::cout << evsTs[i] << " ";
        for(std::size_t j = i+1; j < evs.size(); j++)
        {
            if(evs[i].v == evs[j].v)// same y value
            {
                if(evs[i].u == evs[j].u+1)
                {
                    dtx += (evsTs[i] - evsTs[j]);
                    // std::cout << "1) " << (evsTs[i] - evsTs[j]) << std::endl;
                    nx++;
                }
                else if(evs[i].u == evs[j].u-1)
                {
                    dtx -= (evsTs[i] - evsTs[j]);
                    // std::cout << "2) " << (evsTs[i] - evsTs[j]) << std::endl;
                    nx++;
                }    
            }
            if(evs[i].u == evs[j].u)// same x value
            {
                if(evs[i].v == evs[j].v+1)
                {
                    dty += (evsTs[i] - evsTs[j]);
                    // std::cout << "3) " << (evsTs[i] - evsTs[j]) << std::endl;
                    ny++;
                }
                    
                else if(evs[i].v == evs[j].v-1)
                {
                    dty -= (evsTs[i] - evsTs[j]);
                    // std::cout << "4) " << (evsTs[i] - evsTs[j]) << std::endl;
                    ny++;
                }
            }
        }
    }
    if(nx || ny)
    {
        double eps = 1e-10;
        double tx, ty;
        if(nx) tx = dtx/nx;
        else tx = 0;
        if(ny) ty = dty/ny;
        else ty = 0;
        double norm = 1/(tx*tx + ty*ty);
        vel[handL].u = tx*norm;
        vel[handL].v = ty*norm;
        // double norm = 1/(dtx*dtx + dty*dty + eps)*(1/nx + 1/ny + eps);
        // vel[handL].u = dtx*norm;
        // vel[handL].v = dty*norm;
        double lim = 10000;
        if(std::fabs(vel[handL].u) > lim) vel[handL].u = 0;
        if(std::fabs(vel[handL].v) > lim) vel[handL].v = 0;
        // std::cout << evsTs[evs.size()-1] << ") "<< vel[handL].u << " / " <<  vel[handL].v << "\t-> " << nx << " - " << ny << std::endl;
        // std::cout << vel[handL].u << " " << vel[handL].v << std::endl;
        // std::cout << std::endl;
    }
    
    return vel;
    */

   sklt vel;
    for (size_t i = 0; i < vel.size(); i++)
    {
        vel[i].u = 0;
        vel[i].v = 0;
    }
    // calculate velocity for hand L joint
    double dtx = 0.0, dty = 0.0, k=1e2;
    long int nx = 0, ny = 0;
    for(std::size_t i = 0; i < evs.size()-1; i++)
    {
        // std::cout << evsTs[i] << " ";
        for(std::size_t j = i+1; j < evs.size(); j++)
        {
            if(evs[i].v == evs[j].v)// same y value
            {
                if(evs[i].u == evs[j].u+1)
                {
                    dtx += (evsTs[i] - evsTs[j]);
                    nx++;
                }
                else if(evs[i].u == evs[j].u-1)
                {
                    dtx -= (evsTs[i] - evsTs[j]);
                    nx++;
                }    
            }
            if(evs[i].u == evs[j].u)// same x value
            {
                if(evs[i].v == evs[j].v+1)
                {
                    dty += (evsTs[i] - evsTs[j]);
                    ny++;
                }
                    
                else if(evs[i].v == evs[j].v-1)
                {
                    dty -= (evsTs[i] - evsTs[j]);
                    ny++;
                }
            }
        }
    }
    double eps = 1e-10;
    double norm = 1/(dtx*dtx + dty*dty + eps);
    vel[jointName].u = dtx*norm/nx*k;
    vel[jointName].v = dty*norm/ny*k;
    // double lim = 100;
    // if(std::fabs(vel[jointName].u) > lim) vel[jointName].u = 0;
    // if(std::fabs(vel[jointName].v) > lim) vel[jointName].v = 0;
    // std::cout << vel[elbowL].u << " " << vel[elbowL].v << std::endl;
    // // std::cout << std::endl;
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