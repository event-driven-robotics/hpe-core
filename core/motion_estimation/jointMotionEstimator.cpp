
#include "jointMotionEstimator.h"

using namespace hpecore;


sklt jointMotionEstimator::resetPose(sklt detection)
{
    return detection;
}

sklt jointMotionEstimator::resetVel()
{
    sklt vel;
    for (size_t i = 0; i < vel.size(); i++)
    {
        vel[i].u = 0;
        vel[i].v = 0;
    }
    return vel;
}

sklt jointMotionEstimator::estimateVelocity(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int k, sklt dpose, std::deque<joint>& vels)
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
    int delta = 1;
    // for(std::size_t i = 0; i < evs.size()-1; i++)
    for(std::size_t i = 0; i < k-1; i++) // new events
    {
        // std::cout << i << ")) " << std::endl;
        // variables for event-by-event velocities
        double tx = 0, ty = 0;
        int nxE = 0, nyE = 0;
        // for(std::size_t j = i+1; j < evs.size(); j++)
        for(std::size_t j = k; j < evs.size(); j++) // past events
        {
            if(evs[i].v != evs[j].v || evs[i].u != evs[j].u)
            {
                if(evs[i].v == evs[j].v) // same y value
                {
                    if(evs[i].u == evs[j].u + delta)
                    {
                        dtx += (evsTs[i] - evsTs[j]);
                        nx++;
                        tx = (evsTs[i] - evsTs[j]);
                        nxE++;
                        // std::cout << j << ") A [" << evs[i].u << ", "<< evs[i].v
                        // << "] - [" << evs[j].u << ", " << evs[j].v << "]" << std::endl;
                    }
                    else if(evs[i].u == evs[j].u-delta)
                    {
                        dtx -= (evsTs[i] - evsTs[j]);
                        nx++;
                        tx = -(evsTs[i] - evsTs[j]);
                        nxE++;
                        // std::cout << j << ") B [" << evs[i].u << ", "<< evs[i].v
                        // << "] - [" << evs[j].u << ", " << evs[j].v << "]" << std::endl;
                    }    
                }
                if(evs[i].u == evs[j].u) // same x value
                {
                    if(evs[i].v == evs[j].v+delta)
                    {
                        dty += (evsTs[i] - evsTs[j]);
                        ny++;
                        ty = (evsTs[i] - evsTs[j]);
                        nyE++;
                        // std::cout << j << ") C [" << evs[i].u << ", "<< evs[i].v
                        // << "] - [" << evs[j].u << ", " << evs[j].v << "]" << std::endl;
                    }
                        
                    else if(evs[i].v == evs[j].v-delta)
                    {
                        dty -= (evsTs[i] - evsTs[j]);
                        ny++;
                        ty = -(evsTs[i] - evsTs[j]);
                        nyE++;
                        // std::cout << j << ") D [" << evs[i].u << ", "<< evs[i].v
                        // << "] - [" << evs[j].u << ", " << evs[j].v << "]" << std::endl;
                    }
                }
            }
        }
        // std::cout << std::endl;
        // calculate and save event-by-event velocities
        if(nxE && nyE)
        {
            double epsE = 1e-20;
            double normE = 1/(tx*tx + ty*ty+epsE);
            double vx = tx*normE/(nxE+epsE);
            double vy = ty*normE/(nyE+epsE);
            joint V {vx, vy};
            vels.push_back(V);
        }

    }

    // Calculate and save by-batch velocities

    // Franco's way
    double eps = 0;
    double norm = 1/(dtx*dtx + dty*dty+eps);
    vel[jointName].u = dtx*norm/(nx+eps);
    vel[jointName].v = dty*norm/(ny+eps);
    // double lim =1e2;
    // if(std::fabs(vel[jointName].u) > lim) vel[jointName].u = 0;
    // if(std::fabs(vel[jointName].v) > lim) vel[jointName].v = 0;

    // Arren'sway
    // if(nx == 0 && ny ==0) {
    //     dtx = 1e6; dty = 1e6;
    // }
    // if(nx) dtx /= nx;
    // if(ny) dty /= ny;

    // double dtdp = sqrt(dtx*dtx + dty*dty);
    // double speed = 1.0 / dtdp;
    // double angle = atan2(dty, dtx);
    // vel[jointName].u = speed * cos(angle);
    // vel[jointName].v = speed * sin(angle);
    
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

void jointMotionEstimator::estimateFire(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt pose, sklt dpose, double **Te, cv::Mat matTe)
{
    double du = dpose[jointName].u;
    double dv = dpose[jointName].v;
    double den = du*du + dv*dv + 1e-10;
    double duI = du/den;
    double dvI = dv/den;

    int dimY = 240, dimX = 346;
    // for (int i = 0; i < dimY; i++) 
    // {
    //     for (int j = 0; j < dimX; j++) 
    //         Te[i][j] = 0 ;
    // }

    // Use actual events to compute Te
    /*
    for(std::size_t k = nevs; k < evs.size(); k++)
    
    {
        int x = evs[k].u;
        int y = evs[k].v;
        double ts = evsTs[k];
        // // int value = evsPol[i];
        for (int i = 0; i < dimY; i++) 
        {
            for (int j = 0; j < dimX; j++) 
            {
                double prod = duI * (j-x) + dvI * (i-y) + ts;
                // if(prod > Te[i][j] || fabs(Te[i][j]) > 1e5)
                if(prod > 0)
                {
                    Te[i][j] = prod;
                }
            }
        }
    }
    // for (int i = 0; i < dimY; i++) 
    // {
    //     for (int j = 0; j < dimX; j++) 
    //         Te[i][j] = Te[i][j]/(evs.size()-nevs);
    // }
    */


    // Test: create simple edge to check the gradient
    std::deque<joint> evs2;
    int x0 = pose[jointName].u;
    int y0 = pose[jointName].v;
    for(int i=x0-10; i<x0+10; i++)
    {
        joint J = {i, y0};
        evs2.push_back(J);
    }
    for(int i=y0-10; i<y0+10; i++)
    {
        joint J = {x0, i};
        evs2.push_back(J);
    }
    // fisrt set to zero all the rest of the Te matrix
    for (int i = 0; i < dimY; i++) 
    {
        for (int j = 0; j < dimX; j++) 
            Te[i][j] = 0 ;
    }
    
    double modV = sqrt(du*du + dv*dv);
    for(std::size_t k = 0; k < evs.size(); k++) // to show fake events use evs2 instead of evs
    {
        int x = evs[k].u;
        int y = evs[k].v;
        double ts = evsTs[0];

        for (int i = 0; i < dimY; i++) 
        {
            for (int j = 0; j < dimX; j++) 
            {
                double prod = duI * (j-x) + dvI * (i-y);
                double newTe = 1/modV * sqrt((i-y)*(i-y) + (j-x)*(j-x));
                prod = newTe;
                double a = ((j-x)/du);
                double b = ((i-y)/dv);
                double eps = 1.5;
                // std::cout << a << " " << b << std::endl;
                // if(a == b || (a > b-eps && a < b+eps))
                if((prod < Te[i][j] || Te[i][j] == 0) && (a == b || (a > b-eps && a < b+eps)) && (j-x)*du>0)
                {
                    Te[i][j] = prod;
                }
            }
        }
        
    }
    // search max
    double max = Te[0][0];
    for (int i = 0; i < dimY; i++) 
    {
        for (int j = 0; j < dimX; j++)
        {
            if(Te[i][j] > max)
                max = Te[i][j];
        }  
    }
    // substract max where non zero
    for (int i = 0; i < dimY; i++) 
    {
        for (int j = 0; j < dimX; j++)
        {
            if(Te[i][j] > 0)
                Te[i][j] = max - Te[i][j];
        }  
    }
    // cv mat
    for (int i = 0; i < dimY-1; i++) 
    {
        for (int j = 0; j < dimX-1; j++) 
        {
            matTe.at<float>(i, j) = float(Te[i][j]);
        }
    }
    cv::normalize(matTe,matTe, 0, 1, cv::NORM_MINMAX);

}

double jointMotionEstimator::getError(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt pose, sklt dpose, double **Te, cv::Mat matTe)
{
    double sum = 0;
    int cont = 0;
    int dimY = 240, dimX = 346;

    int end = (nevs < int(evs.size())) ? nevs : evs.size();
    for(int k = 0; k < end; k++)
    {
        int x = evs[k].u;
        int y = evs[k].v;
        double ts = evsTs[k];
        if(x >= 0 && x<dimX && y >= 0 && y<dimY)
        {
            double err = Te[y][x] - ts;
            sum += err;
            cont++;
        }
    }
    if(cont)
        return sum/cont;
    else
        return 0;
}

joint jointMotionEstimator::centerMass(std::deque<joint> evs,  int jointName, int nevs, sklt pose)
{
    double sumX = 0, sumY = 0;
    int cont = 0;
    int dimY = 240, dimX = 346;
    double X, Y;
    joint center = {0, 0};


    int end = (nevs < int(evs.size())) ? nevs : evs.size();
    for(int k = 0; k < end; k++)
    {
        int x = evs[k].u;
        int y = evs[k].v;
        if(x >= 0 && x<dimX && y >= 0 && y<dimY)
        {
            sumX += x;
            sumY += y;
            cont++;
        }
    }
    if(cont) 
       center = {sumX/cont, sumY/cont};

    return center;    
}

sklt jointMotionEstimator::setVel(int jointName, sklt dpose, double dx, double dy, double err)
{ 
    sklt vel;
    for (size_t i = 0; i < vel.size(); i++)
    {
        vel[i].u = 0;
        vel[i].v = 0;
    }

    // double delta = dx + dy;
    // double dxp = dx/delta;
    // double dyp = dy/delta;
    // vel[jointName].u = err *  dxp;
    // vel[jointName].v = err * dyp;

    vel[jointName].u = err/2;
    vel[jointName].v = err/2;

    return vel;
}


sklt jointMotionEstimator::nearestEvent(std::deque<joint> evs, std::deque<double> evsTs, int jointName, int nevs, sklt dpose, std::deque<joint>& vels)
{
    double vx = 0, vy = 0;
    int num = 0;

    sklt vel;
    for (size_t i = 0; i < vel.size(); i++)
    {
        vel[i].u = 0;
        vel[i].v = 0;
    }

    // erase repeated events from the past
    for(size_t i=nevs-1; i < evs.size()-1; i++)
	{
		for(size_t j=i+1; j<evs.size(); j++)
		{
			if(evs[i].u == evs[j].u && evs[i].v == evs[j].v)
                evs.erase(evs.begin()+j);
		}
	}

    // Estimation using just the single nearest event from the past
    // if(nevs < int(evs.size())) // there are past events
    // {   
    //     for(std::size_t i = 0; i < nevs-1; i++) // new events
    //     {
    //         int x = evs[i].u;
    //         int y = evs[i].v;
    //         double ts = evsTs[i];
    //         double distMin;
    //         int jmin;
    //         for(std::size_t j = nevs; j < evs.size(); j++)   // past events 
    //         {
    //             int xp = evs[j].u;
    //             int yp = evs[j].v;
    //             double tsp = evsTs[j];
    //             double dist = sqrt((xp-x)*(xp-x) + (yp-y)*(yp-y));
    //             if(j == nevs || (dist < distMin && dist > 0))
    //             {
    //                 distMin = dist;
    //                 jmin = j;
    //             }  
    //         }
    //         // event-by-event  velocity
    //         double vxE = (x - evs[jmin].u)/(ts - evsTs[jmin]);
    //         double vyE = (y - evs[jmin].v)/(ts - evsTs[jmin]);
    //         vx += vxE;
    //         vy += vyE;
    //         joint V {vxE, vyE};
    //         vels.push_back(V);
    //         // accumulated velocity
    //         vx += (x - evs[jmin].u)/(ts - evsTs[jmin]);
    //         vy += (y - evs[jmin].v)/(ts - evsTs[jmin]); 
    //         num++;
    //     }
    //     if(num) // average velocity
    //     {
    //         vx /= num;
    //         vy /= num;
    //     }
    // }

    // Estimation using more than one event from thepast
    if(nevs < int(evs.size())) // there are  past events
    {
        for(std::size_t i = 0; i < nevs-1; i++) // new events
        {
            int x = evs[i].u;
            int y = evs[i].v;
            double ts = evsTs[i];
            double rad = 2.97;
            double vxE = 0, vyE = 0;
            int numE = 0;
            for(std::size_t j = nevs; j < evs.size(); j++) // past events
            {
                int xp = evs[j].u;
                int yp = evs[j].v;
                double tsp = evsTs[j];
                double dist = sqrt((xp-x)*(xp-x) + (yp-y)*(yp-y));
                double dt = ts-tsp;
                if(dist <= rad && dist > 0)
                // if(fabs(x-xp) <= rad && fabs(y-yp) <= rad)
                {
                    vxE += (x-xp)/dt;
                    vyE += (y-yp)/dt;
                    numE++;
                }  
            }
            // event-by-event  velocity
            if(numE)
            {
                vxE /= numE;
                vyE /= numE;
                vx += vxE;
                vy += vyE;
                joint V {vxE, vyE};
                vels.push_back(V);
                num++;
            }
        }
        if(num) // average velocity
        {
            vx /= num;
            vy /= num;
        }
    }

    vel[jointName].u = vx;
    vel[jointName].v = vy;

    return vel;
}