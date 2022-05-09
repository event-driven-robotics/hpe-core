/* BSD 3-Clause License

Copyright (c) 2021, Event Driven Perception for Robotics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */
#pragma once

#include "utility.h"
#include "representations.h"
#include <opencv2/opencv.hpp>

namespace hpecore {

class queuedVelocity
{
private:
    std::deque<pixel_event> q;
    std::deque<pixel_event> qROI[13];
    int roi_width{40};
    int minor_width{3};
    int max_neighbours{3};
    int q_limit{1000};

public:
    void setParameters(int roi_width, int minor_width, int max_neighbours, int q_limit) {
        this->roi_width = roi_width;
        this->minor_width = minor_width;
        this->max_neighbours = max_neighbours;
        this->q_limit = q_limit;
    }

    template <typename T>
    jDot update(const T &q_new, joint j) {
        pixel_event npe;
        int n_added = 0;
        int n_used = 0;
        jDot velocity{0.0, 0.0};

        // see if the new event is in the roi of the joint, add it to the queue,
        // and then calculate a new velocity for the joint
        for (auto &v_new : q_new) {
            // check if it's in the roi
            if (fabs(v_new.x - j.u) < roi_width && fabs(v_new.y - j.v) < roi_width) {
                npe.stamp = v_new.stamp;
                npe.x = v_new.x;
                npe.y = v_new.y;
                q.push_front(npe);
                n_added++;
            } else {
                continue;
            }
            if (n_added)
            {
                // if it was in the roi also compute the velocity
                int n_neighbours = 0;
                double xdot = 0.0, ydot = 0.0;
                for (int i = n_added; i < q.size(); i++) {
                    auto &v_old = q[i];

                    // calculate distance and time
                    int dx = v_new.x - v_old.x;
                    int dy = v_new.y - v_old.y;
                    if (abs(dx) + abs(dy) <= minor_width) {
                        double inv_dt = 1.0 / (v_new.stamp - v_old.stamp);
                        xdot += dx * inv_dt;
                        ydot += dy * inv_dt;
                        n_neighbours++;
                    }
                    if (n_neighbours > max_neighbours) break;
                }

                // calculate the average for this event
                if (n_neighbours) {
                    double inv_neighbours = 1.0 / n_neighbours;
                    velocity.u += (xdot * inv_neighbours);
                    velocity.v += (ydot * inv_neighbours);
                    n_used++;
                }
            }

        }

        // calculate the average velocity over all new events
        if (n_used) {
            double inv_used = 1.0 / n_used;
            velocity.u *= inv_used;
            velocity.v *= inv_used;
        }

        //std::cout << "* " << velocity.u << ", " << velocity.v << std::endl;

        // remove events from the old q
        while (q.size() > q_limit)
            q.pop_back();

        return velocity;
    }


    template <typename T>
    skeleton13_vel update(const T &q_new, skeleton13 state)
    {
        pixel_event npe;
        int n_added[13] = {0};
        int n_used[13] = {0};
        skeleton13_vel velocity = {0};

        // see if the new event is in the roi of each joint, add it to the corresponding queue,
        // and then calculate a new velocity for the joint
        for (auto &v_new : q_new)
        {
            for (int j = 0; j < 13; j++)
            {
                // check if it's in the j-th roi
                if (fabs(v_new.x - state[j].u) < roi_width/2 && fabs(v_new.y - state[j].v) < roi_width/2)
                {
                    npe.stamp = v_new.stamp;
                    npe.x = v_new.x;
                    npe.y = v_new.y;
                    qROI[j].push_front(npe);
                    n_added[j]++;
                }
                if(n_added[j])
                {
                    // if it was in the roi also compute the velocity
                    int n_neighbours = 0;
                    double xdot = 0.0, ydot = 0.0;
                    for (int i = n_added[j]; i < qROI[j].size(); i++)
                    {
                        auto &v_old = qROI[j][i];

                        // calculate distance and time
                        int dx = v_new.x - v_old.x;
                        int dy = v_new.y - v_old.y;
                        if (sqrt(dx*dx + dy*dy)<= minor_width)
                        // if (abs(dx) + abs(dy) <= minor_width) // (F) change to sqrt(dx^2+dy^2)
                        {
                            double inv_dt = 1.0 / (v_new.stamp - v_old.stamp);
                            xdot += dx * inv_dt;
                            ydot += dy * inv_dt;
                            n_neighbours++;
                        }
                        // if (n_neighbours > max_neighbours) break; // (F) what?
                    }

                    // calculate the average for this event
                    if (n_neighbours)
                    {
                        double inv_neighbours = 1.0 / n_neighbours; // (F) why doing this?
                        velocity[j].u += (xdot * inv_neighbours);
                        velocity[j].v += (ydot * inv_neighbours);
                        n_used[j]++;
                    }
                }
            }            
        }

        for (int j = 0; j < 13; j++)
        {
            // calculate the average velocity over all new events
            if (n_used[j])
            {
                double inv_used = 1.0 / n_used[j]; // (F) why doing this?
                velocity[j].u *= inv_used;
                velocity[j].v *= inv_used;
            }

            // remove events from the old q
            while (qROI[j].size() > q_limit)
                qROI[j].pop_back();
        }
    
        return velocity;
    }

};

class surfacedVelocity
{
private:
    std::deque<pixel_event> q;
    std::deque<pixel_event> qROI[13];
    int roi_width{40};
    int minor_width{3};
    int max_neighbours{3};
    int q_limit{1000};
    cv::Mat SAE, SAE_vis;
    hpecore::TOS tos;

public:
    void setParameters(int roi_width, int minor_width, int max_neighbours, int q_limit, cv::Size image_size) {
        this->roi_width = roi_width;
        this->minor_width = minor_width;
        this->max_neighbours = max_neighbours;
        this->q_limit = q_limit;

        this->SAE = cv::Mat::zeros(image_size, CV_32F);
        this->SAE_vis = cv::Mat::zeros(image_size, CV_32F);
        this->tos.init(image_size.width, image_size.height, 7, 2);
    }

    template <typename T>
    skeleton13_vel update(const T &q_new, skeleton13 state)
    {
        int n_used[13] = {0};
        skeleton13_vel velocity = {0};

        for (auto &v_new : q_new) // each event
        {
            for (int k = 0; k < 13; k++) // each joint
            {
                // check if it's in the k-th roi and iff compute the velocity
                if (fabs(v_new.x - state[k].u) < roi_width/2 && fabs(v_new.y - state[k].v) < roi_width/2)
                {
                    int n_neighbours = 0;
                    double xdot = 0.0, ydot = 0.0;
                    for(int i = v_new.x-minor_width; i <= v_new.x+minor_width; i++) // past events
                    {
                        for(int j = v_new.y-minor_width; j <= v_new.y+minor_width; j++)
                        {
                            if(int(this->tos.getSurface().at<unsigned char>(j, i)) && (v_new.x!=i || v_new.y!=j))
                            {
                                double inv_dt = 1.0 / (v_new.stamp - this->SAE.at<float>(j, i));
                                int dx = v_new.x - i;
                                int dy = v_new.y - j;
                                xdot += dx * inv_dt;
                                ydot += dy * inv_dt;
                                n_neighbours++;
                            }
                        }
                    }

                    // calculate the average for this event
                    if (n_neighbours > 1)
                    {
                        double inv_neighbours = 1.0 / n_neighbours; // (F) why doing this?
                        velocity[k].u += (xdot * inv_neighbours);
                        velocity[k].v += (ydot * inv_neighbours);
                        n_used[k]++;
                    }
                }
            }            
        }

        for (int k = 0; k < 13; k++)
        {
            // calculate the average velocity over all new events
            if (n_used[k])
            {
                double inv_used = 1.0 / n_used[k]; // (F) why doing this?
                velocity[k].u *= inv_used;
                velocity[k].v *= inv_used;
            }
        }

        for (auto &qi : q_new)
        {
            this->tos.update(qi.x, qi.y);
            this->SAE.at<float>(qi.y, qi.x) = float(qi.stamp);
        }

      
        return velocity;
    }

template <typename T>
    void errorToVel(const T &q_new, skeleton13 state, skeleton13_vel prevVelocity, std::deque<hpecore::jDot>* error)                                
    {
        for (auto &v_new : q_new) // each event
        {
            for (int k = 0; k < 13; k++) // each joint
            {
                // check if it's in the k-th roi and iff compute the velocity
                if (fabs(v_new.x - state[k].u) < roi_width/2 && fabs(v_new.y - state[k].v) < roi_width/2)
                {
                    int n_neighbours = 0;
                    double xdot = 0.0, ydot = 0.0;
                    jDot vel = {0};
                    for(int i = v_new.x-minor_width; i <= v_new.x+minor_width; i++) // past events
                    {
                        for(int j = v_new.y-minor_width; j <= v_new.y+minor_width; j++)
                        {
                            if(int(this->tos.getSurface().at<unsigned char>(j, i)) && (v_new.x!=i || v_new.y!=j))
                            {
                                double inv_dt = 1.0 / (v_new.stamp - this->SAE.at<float>(j, i));
                                int dx = v_new.x - i;
                                int dy = v_new.y - j;
                                xdot += dx * inv_dt;
                                ydot += dy * inv_dt;
                                n_neighbours++;
                            }
                        }
                    }

                    // calculate the average for this event
                    if (n_neighbours > 1)
                    {
                        double inv_neighbours = 1.0 / n_neighbours; // (F) why doing this?
                        vel.u = (xdot * inv_neighbours);
                        vel.v = (ydot * inv_neighbours);
                        jDot e{vel.u - prevVelocity[k].u, vel.v - prevVelocity[k].v};
                        error[k].push_back(e);
                    }
                }
            }            
        }
        // update time surfaces
        for (auto &qi : q_new)
        {
            this->tos.update(qi.x, qi.y);
            this->SAE.at<float>(qi.y, qi.x) = float(qi.stamp);
        }
    }

    skeleton13_vel updateOnError(skeleton13_vel prevVelocity, std::deque<hpecore::jDot>* error)
    {
        skeleton13_vel velocity = {0};

        for(int j=0; j<13; j++) // each joint
        {
            int n_error = 0;
            double ex = 0.0, ey = 0.0;
            while (!error[j].empty())
            {
                ex += error[j].front().u;
                ey += error[j].front().v;
                error[j].pop_front();
                n_error++;
            }
            if(n_error)
            {
                ex /= n_error;
                ey /= n_error;
                velocity[j].u = prevVelocity[j].u + ex;
                velocity[j].v = prevVelocity[j].v + ey;
            }
        }

        return velocity;
    }

    void getImages(cv::Mat &SAEout, cv::Mat &TOSout)
    {
        cv::normalize(this->SAE,SAEout, 0, 1, cv::NORM_MINMAX); // for visualization purposes
        this->tos.getSurface().copyTo(TOSout);
    }
};
}