
#include "jointMotionEstimator.h"

using namespace hpecore;


sklt jointMotionEstimator::resetPose(sklt detection)
{
    return detection;
}

sklt jointMotionEstimator::estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int k, sklt dpose)
{
    sklt vel;
    for (size_t i = 0; i < vel.size(); i++)
    {
        vel[i].u = 0;
        vel[i].v = 0;
    }
    // erase repeated events
    for(size_t i=0; i < evs.size()-1; i++)
	{
		for(size_t j=i+1; j<evs.size(); j++)
		{
			if(evs[i].u == evs[j].u && evs[i].v == evs[j].v)
                evs.erase(evs.begin()+j);
		}
	}
    // calculate velocity for selected joint
    double dtx = 0.0, dty = 0.0;
    long int nx = 0, ny = 0;
    for(std::size_t i = 0; i < evs.size()-1; i++)
    {
        // std::cout << evsTs[i] << " ";
        for(std::size_t j = i+1; j < evs.size(); j++)
        {
            if(evs[i].v != evs[j].v || evs[i].u != evs[j].u)
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
    }
    
    if(nx == 0 && ny ==0) {
        dtx = 1e6; dty = 1e6;
    }
    if(nx) dtx /= nx;
    if(ny) dty /= ny;

    double dtdp = sqrt(dtx*dtx + dty*dty);
    double speed = 1.0 / dtdp;
    double angle = atan2(dty, dtx);
    vel[jointName].u = speed * cos(angle);
    vel[jointName].v = speed * sin(angle);
    //double eps = 1e-20;
    // double norm = 1/(dtx*dtx + dty*dty+eps);
    // vel[jointName].u = dtx*norm/(nx+eps)*k;
    // vel[jointName].v = dty*norm/(ny+eps)*k;
    // double lim =1e2;
    // if(std::fabs(vel[jointName].u) > lim) vel[jointName].u = 0;
    // if(std::fabs(vel[jointName].v) > lim) vel[jointName].v = 0;
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
