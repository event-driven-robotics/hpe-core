
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
    // double eps = 1e-20;
    double norm = 1/(dtx*dtx + dty*dty);
    vel[jointName].u = dtx*norm/nx*k;
    vel[jointName].v = dty*norm/ny*k;
    double lim = 2e2;
    // if(std::fabs(vel[jointName].u) > lim) vel[jointName].u = 0;
    // if(std::fabs(vel[jointName].v) > lim) vel[jointName].v = 0;

    // if(vel[jointName].u > lim) vel[jointName].u = dpose[jointName].u + 10;
    // if(vel[jointName].u < -lim) vel[jointName].u = dpose[jointName].u - 10;
    // if(vel[jointName].v > lim) vel[jointName].v = dpose[jointName].v + 10;
    // if(vel[jointName].v < -lim) vel[jointName].v = dpose[jointName].v - 10;

    // if(std::fabs(vel[jointName].u) > std::fabs(dpose[jointName].u)) vel[jointName].u = dpose[jointName].u*1.1;
    // if(std::fabs(vel[jointName].v) > std::fabs(dpose[jointName].v)) vel[jointName].v = dpose[jointName].v*1.1;

    if(std::fabs(vel[jointName].u) > lim) vel[jointName].u = dpose[jointName].u*1.01;
    if(std::fabs(vel[jointName].v) > lim) vel[jointName].v = dpose[jointName].v*1.01;


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

using namespace cv;
using namespace std;

cv::Mat jointMotionEstimator::projNew(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt dpose, double tnow)
{
    cv::Mat test = cv::Mat::zeros(cv::Size(346, 240), CV_32FC1);
    double vx = dpose[jointName].u;
    double vy = dpose[jointName].v;

    // new events (blue)
    int end = (nevs < int(evs.size())) ? nevs : evs.size();
    for(int i = 0; i < end; i++)
    // for(std::size_t i = 0; i < nevs; i++)
    {
        // int value = evsPol[i];
        if(evs[i].u<346 && evs[i].v<240)
        {   
            // int x = evs[i].u;
            // int y = evs[i].v;
            int dx = (int)round(vx * (tnow- evsTs[i]) * 1);
            int dy = (int)round(vy * (tnow- evsTs[i]) * 1);
            int x = evs[i].u - dx;
            int y = evs[i].v - dy;
            if(abs(x)<346 && abs(y)<240)
                test.at<float>(y, x) = 1.0f;
        }
    }

    return test;
}

cv::Mat jointMotionEstimator::projPrev(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt dpose, double tnow)
{
    cv::Mat test = cv::Mat::zeros(cv::Size(346, 240), CV_32FC1);
    double vx = dpose[jointName].u;
    double vy = dpose[jointName].v;
  
    for(std::size_t i = nevs; i < evs.size(); i++)
    {
        // int value = evsPol[i];
        if(evs[i].u<346 && evs[i].v<240)
        {   
            int dx = (int)round(vx * (tnow- evsTs[i]) * 1);
            int dy = (int)round(vy * (tnow- evsTs[i]) * 1);
            int x = evs[i].u + dx;
            int y = evs[i].v + dy;
            if(x>=0 && x<=346 && y>=0 && y<=240)
                test.at<float>(y, x) = 1.0f;
        }
    }
 
    return test;
}


sklt jointMotionEstimator::comparteProjs(int jointName, sklt dpose, cv::Mat newEvs, cv::Mat oldEvs)
{
    sklt vel;
    for (size_t i = 0; i < vel.size(); i++)
    {
        vel[i].u = 0;
        vel[i].v = 0;
    }

    for(int i = 0; i < newEvs.cols; i++)
    {
        for(int j = 0; j < newEvs.rows; j++)
        {
            // compare pixels
            // if(newEvs.at<float>(y, x) == 1 && newEvs.at<float>(y, x) != 1)
                
        }
    }

    return vel;
}

cv::Mat jointMotionEstimator::estimateFire(std::deque<joint> evs, std::deque<double> evsTs, std::deque<int> evsPol, int jointName, int nevs, sklt pose, sklt dpose, double tnow, double **Te, cv::Mat matTe)
{
    // cv::Mat test = cv::Mat::zeros(cv::Size(346, 240), CV_32FC1);
    double u = pose[jointName].u;
    double v = pose[jointName].v;
    double du = dpose[jointName].u;
    double dv = dpose[jointName].v;

    double den = du*du+dv*dv;
    double duI = du/den;
    double dvI = -dv/den;

    int dimY = 240, dimX = 346;
    
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
                // double newTe = duI * (i-y) + dvI * (j-x) + ts;
                double newTe = sqrt(duI *(i-y)*(i-y) + dvI *(j-x)*(j-x))+ ts;
                // Te[i][j] = newTe
                // if(newTe > 24) newTe = 24;
                // if(newTe < Te[i][j]) Te[i][j] = newTe;
                if(newTe < Te[i][j])
                {
                    Te[i][j] = newTe;
                    // std::cout << Te[i][j] << std::endl;
                    // if(Te[i][j] < 1)
                        
                    // else 
                    //     matTe.at<float>(i, j) = 0;
                }
                else 
                    Te[i][j] = 1e20;
                if(Te[i][j] < 100)
                {
                    double aux = (Te[i][j]-ts);
                    if(evsTs[k] < Te[i][j])
                    {
                        matTe.at<float>(i, j) = 1 - aux;
                        std::cout << evsTs[k] << " - " << Te[i][j] < std::endl;
                    }
                    // std::cout << aux << std::endl;
                }
                else 
                matTe.at<float>(i, j) = 0.;
                // else 
                //     matTe.at<float>(i, j) = 0;
                // if(newTe > 24) newTe = 24;
                // else if(newTe < 0) newTe = 0;
                // if(newTe < Te[j][i])
                    // Te[j][i] = newTe;
                // if(x>=0 && x<=346 && y>=0 && y<=240)
                    // matTe.at<float>(i, j) = 1 - (Te[i][j]-ts);
                    
            }
        }
    }


    // for(std::size_t k = nevs; k < evs.size(); k++)
    // {
    //     int x = evs[k].u;
    //     int y = evs[k].v;
    //     double ts = evsTs[k];
    //     for (int i = 0; i < dimY; i++) 
    //     {
    //         for (int j = 0; j < dimX; j++) 
    //         // {
    //         {
    //             // double newTe = duI * (i-x) + dvI * (j-y) + ts;
    //             //if(newTe < Te[i][j])
    //                 // Te[j][i] = sqrt((i-x)*(i-x) + (j-y)*(j-y));
    //             //Te[i][j] = i/100.0;;
    //             if(x>=0 && x<=346 && y>=0 && y<=240)
    //             //     // matTe.at<float>(y, x) = 1 - (duI * (i-x) + dvI * (j-y))/20;
    //                 matTe.at<float>(y, x) = 0.7;
    //         }
    //     }
    //     // int value = evsPol[i];
    //     // if(evs[i].u<346 && evs[i].v<240)
    //     // {   
    //     //     // if(x>=0 && x<=346 && y>=0 && y<=240 && ts > Te[y][x])
    //     //     //     test.at<float>(y, x) = 1.0f;
    //     // }
    // }
    // for (int i = 0; i < dimY; i++) 
    // {
    //     for (int j = 0; j < dimX; j++)
    //     {
    //         matTe.at<float>(j, i) = Te[i][j]/346;
    //         // std::cout << i << ", " << j << " -> " << Te[i][j] << std::endl;
    //     }
    // }
 
    return matTe;
}