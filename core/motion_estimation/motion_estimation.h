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
#include <opencv2/opencv.hpp>
namespace hpecore {

class queuedVelocity
{
private:
    std::deque<pixel_event> q;
    int roi_width{40};
    int minor_width{3};
    int max_neighbours{3};
    int q_limit{1000};

public:

    void setParameters(int roi_width, int minor_width, int max_neighbours, int q_limit)
    {
        this->roi_width = roi_width;
        this->minor_width = minor_width;
        this->max_neighbours = max_neighbours;
        this->q_limit = q_limit;
    }

    template<typename T>
    joint update(const T &q_new, joint j) 
    {
        pixel_event npe;
        int n_added = 0;
        int n_used = 0;
        joint velocity{0.0, 0.0};

        // see if the new event is in the roi of the joint, add it to the queue,
        // and then calculate a new velocity for the joint
        for(auto &v_new : q_new) 
        {
            //check if it's in the roi
            if (fabs(v_new.x - j.u) < roi_width && fabs(v_new.y - j.v) < roi_width) {
                npe.stamp = v_new.stamp; npe.x = v_new.x; npe.y = v_new.y;
                q.push_front(npe);
                n_added++;
            } else {
                continue;
            }

            //if it was in the roi also compute the velocity
            int n_neighbours = 0;
            double xdot = 0.0, ydot = 0.0;
            for(int i = n_added; i < q.size(); i++)
            {
                auto &v_old = q[i];

                //calculate distance and time
                int dx = v_new.x - v_old.x;
                int dy = v_new.y - v_old.y;
                if(abs(dx)+abs(dy) <= minor_width)
                {
                    double inv_dt = 1.0 / (v_new.stamp - v_old.stamp);
                    xdot += dx * inv_dt;
                    ydot += dy * inv_dt;
                    n_neighbours++;
                } 
                if(n_neighbours > max_neighbours) break;
            }

            // calculate the average for this event
            if(n_neighbours)
            {
                double inv_neighbours = 1.0/ n_neighbours;
                velocity.u += (xdot * inv_neighbours);
                velocity.v += (ydot * inv_neighbours);
                n_used++;
            }

        }

        //calculate the average velocity over all new events
        if(n_used) {
            double inv_used = 1.0 / n_used;
            velocity.u *= inv_used;
            velocity.v *= inv_used;
        }

        //remove events from the old q
        while(q.size() > q_limit)
            q.pop_back();

        yInfo() << q.size() << n_used;

        return velocity;
    }




};

}