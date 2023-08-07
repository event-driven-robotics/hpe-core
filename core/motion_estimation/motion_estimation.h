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
    skeleton13_vel update(const T &begin, const T &end, skeleton13 state, double evTs)
    {
        pixel_event npe;
        int n_added[13] = {0};
        int n_used[13] = {0};
        skeleton13_vel velocity = {0};
        int roi_width_half = roi_width/2;

        // see if the new event is in the roi of each joint, add it to the corresponding queue,
        // and then calculate a new velocity for the joint
        for(auto v_new = begin; v_new != end; v_new++) // each event
        {
            for (int j = 0; j < 13; j++)
            {
                // check if it's in the j-th roi
                if (fabs(v_new->x - state[j].u) < roi_width_half && fabs(v_new->y - state[j].v) < roi_width_half)
                {
                    npe.stamp = evTs; // npe.stamp = v_new.stamp;
                    npe.x = v_new->x;
                    npe.y = v_new->y;
                    qROI[j].push_front(npe);
                    n_added[j]++;
                } else {
                    continue;
                }

                // if it was in the roi also compute the velocity
                int n_neighbours = 0;
                double xdot = 0.0, ydot = 0.0;
                for (int i = n_added[j]; i < qROI[j].size(); i++) {
                    auto &v_old = qROI[j][i];

                    // calculate distance and time
                    int dx = v_new->x - v_old.x;
                    int dy = v_new->y - v_old.y;
                    if (abs(dx) + abs(dy) <= minor_width)
                    {
                        double inv_dt = 1.0 / (evTs - v_old.stamp); //double inv_dt = 1.0 / (v_new.stamp - v_old.stamp);
                        xdot += dx * inv_dt;
                        ydot += dy * inv_dt;
                        n_neighbours++;
                    }
                    //if (n_neighbours > max_neighbours) break;
                }

                // calculate the average for this event
                if (n_neighbours) {
                    double inv_neighbours = 1.0 / n_neighbours;
                    velocity[j].u += (xdot * inv_neighbours);
                    velocity[j].v += (ydot * inv_neighbours);
                    n_used[j]++;
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
    cv::Mat SAE, SAE_vis, eros;
    hpecore::EROS tos;

public:
    void setParameters(int roi_width, int minor_width, int max_neighbours, int q_limit, cv::Size image_size) {
        this->roi_width = roi_width;
        this->minor_width = minor_width;
        this->max_neighbours = max_neighbours;
        this->q_limit = q_limit;

        this->SAE = cv::Mat::zeros(image_size, CV_64F);
        // this->SAE_vis = cv::Mat::zeros(image_size, CV_64F);
        this->tos.init(image_size.width, image_size.height, 7, 3);
        eros = cv::Mat(image_size, CV_64F);
        SAE_vis = cv::Mat::zeros(image_size, CV_64F);
    }

    template <typename T>
    skeleton13_vel update(const T &begin, const T &end, skeleton13 state, double evTs)
    {
        int n_used[13] = {0};
        skeleton13_vel velocity = {0};

        for(auto v_new = begin; v_new != end; v_new++) // each event
        {
            for (int k = 0; k < 13; k++) // each joint
            {
                // check if it's in the k-th roi and iff compute the velocity
                if (fabs(v_new->x - state[k].u) < roi_width/2 && fabs(v_new->y - state[k].v) < roi_width/2)
                {
                    int n_neighbours = 0;
                    double xdot = 0.0, ydot = 0.0;
                    for(int i = v_new->x-minor_width; i <= v_new->x+minor_width; i++) // past events
                    {
                        for(int j = v_new->y-minor_width; j <= v_new->y+minor_width; j++)
                        {
                            if(int(this->tos.getSurface().at<unsigned char>(j, i)) && (v_new->x!=i || v_new->y!=j))
                            {
                                double inv_dt = 1.0 / (evTs - this->SAE.at<float>(j, i));
                                int dx = v_new->x - i;
                                int dy = v_new->y - j;
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

        for(auto qi = begin; qi != end; qi++)
        {
            this->tos.update(qi->x, qi->y);
            this->SAE.at<float>(qi->y, qi->x) = float(evTs);
        }

      
        return velocity;
    }

    template <typename T>
    void errorToVel(const T &begin, const T &end, skeleton13 state, skeleton13_vel prevVelocity, std::deque<hpecore::jDot>* error, double evTs)                                
    {
        for(auto v_new = begin; v_new != end; v_new++) // each event
        {
            for (int k = 0; k < 13; k++) // each joint
            {
                // check if it's in the k-th roi and iff compute the velocity
                if (fabs(v_new->x - state[k].u) < roi_width/2 && fabs(v_new->y - state[k].v) < roi_width/2)
                {
                    int n_neighbours = 0;
                    double xdot = 0.0, ydot = 0.0;
                    jDot W = {0};
                    for(int i = v_new->x-minor_width; i <= v_new->x+minor_width; i++) // past events
                    {
                        for(int j = v_new->y-minor_width; j <= v_new->y+minor_width; j++)
                        {
                            if(int(this->tos.getSurface().at<unsigned char>(j, i)) && (v_new->x!=i || v_new->y!=j))
                            {
                                double inv_dt = 1.0 / (evTs - this->SAE.at<float>(j, i));
                                int dx = v_new->x - i;
                                int dy = v_new->y - j;
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
                        W.u = (xdot * inv_neighbours);
                        W.v = (ydot * inv_neighbours);
                        jDot e{W.u - prevVelocity[k].u, W.v - prevVelocity[k].v};
                        error[k].push_back(e);
                    }
                }
            }            
        }
        // update time surfaces
        for(auto qi = begin; qi != end; qi++)
        {
            this->tos.update(qi->x, qi->y);
            this->SAE.at<float>(qi->y, qi->x) = float(evTs);
        }
    }


    template <typename T>
    void errorToCircle(const T &begin, const T &end, skeleton13 state, skeleton13_vel V, std::deque<hpecore::jDot>* error, double evTs)                                
    {
        for(auto v_new = begin; v_new != end; v_new++) // each event
        {
            for (int k = 0; k < 13; k++) // each joint
            {
                // check if it's in the k-th roi and iff compute the velocity
                if (fabs(v_new->x - state[k].u) < roi_width/2 && fabs(v_new->y - state[k].v) < roi_width/2)
                {
                    int n_neighbours = 0;
                    double xdot = 0.0, ydot = 0.0;
                    jDot W = {0};
                    for(int i = v_new->x-minor_width; i <= v_new->x+minor_width; i++) // past events
                    {
                        for(int j = v_new->y-minor_width; j <= v_new->y+minor_width; j++)
                        {
                            if(int(this->tos.getSurface().at<unsigned char>(j, i)) && (v_new->x!=i || v_new->y!=j))
                            {
                                double inv_dt = 1.0 / (evTs - this->SAE.at<float>(j, i));
                                int dx = v_new->x - i;
                                int dy = v_new->y - j;
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
                        W.u = (xdot * inv_neighbours);
                        W.v = (ydot * inv_neighbours);
                        double K = 1 - sqrt((((V[k].u/2)*(V[k].u/2) + (V[k].v/2)*(V[k].v/2)) / ((W.u-V[k].u/2)*(W.u-V[k].u/2) + (W.v-V[k].v/2)*(W.v-V[k].v/2))));
                        jDot e{K*(W.u - V[k].u/2), K*(W.v - V[k].v/2)};
                        error[k].push_back(e);
                    }
                }
            }            
        }
        // update time surfaces
        for(auto qi = begin; qi != end; qi++)
        {
            this->tos.update(qi->x, qi->y);
            this->SAE.at<float>(qi->y, qi->x) = float(evTs);
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

    

    const cv::Mat& querySAE()
    {
        // // cv::Mat SAEout;
        cv::normalize(this->SAE, SAE_vis, 0, 1, cv::NORM_MINMAX);
        // return SAE_vis;
        // return this->SAE;
        cv::Mat SAEout;
        SAE_vis.convertTo(SAEout, CV_8U);
        return SAE_vis;
        // cv::normalize(this->SAE, SAEout, 0, 1, cv::NORM_MINMAX);
        // return SAEout;

        // cv::Mat sae = cv::Mat::zeros(this->SAE.size(), CV_64F);
        // return this->tos.getSurface();
    } 

    const cv::Mat& queryEROS()
    {
        // cv::Mat TOSout;
        this->tos.getSurface().copyTo(eros);
        return eros;
    }
};

class pwvelocity 
{
private:

    //parameters
    double filter_thresh{0.0}; //seconds
    int eros_k{7};
    double eros_d{0.3};

    //internal structure/variables
    cv::Mat eros;
    cv::Mat sae;
    cv::Rect roi_full;
    double ts_prev{0.0}, ts_curr{0.0};
    bool use_norm{false}; //invert vector magnitude
    bool use_om{false};   //observation model

public:
    void setParameters(cv::Size image_size, int eros_k = 7, double eros_d = 0.3, double filter_threshold = 0) 
    {
        eros = cv::Mat(image_size, CV_64F);
        sae =  cv::Mat(image_size, CV_64F);
        roi_full = cv::Rect(cv::Point(0, 0), image_size);
        filter_thresh = filter_threshold;
        this->eros_k = eros_k % 2 ? eros_k : eros_k + 1;
        this->eros_d = eros_d >= 1.0 ? 0.99 : eros_d;
        this->eros_d = eros_d < 0.0 ? 0.01 : eros_d;

    }

    template<typename T>
    void update(const T &begin, const T &end)
    {
        //these are static so they get set once and we use the same memory
        //locations each call
        static double odecay = pow(eros_d, 1.0 / eros_k);
        static int half_kernel = eros_k / 2;
        static cv::Rect roi_raw = cv::Rect(0, 0, eros_k, eros_k);

        for(auto v = begin; v != end; v++)
        {
            double ts = v.timestamp();
            
            //get references to the indexed pixel locations
            auto &p_sae  =  sae.at<double>(v->y, v->x);
            auto &p_eros = eros.at<double>(v->y, v->x);

            {
                auto xl = std::max(v->x - 1, 0);
                auto xh = std::min(v->x + 1 + 1, eros.cols);  //+1 becuase I use < sign
                auto yl = std::max(v->y - 1, 0);
                auto yh = std::min(v->y + 1 + 1, eros.rows);  //+1 becuase I use < sign

                bool add = false;
                for (auto xi = xl; xi < xh; ++xi) {
                    for (auto yi = yl; yi < yh; ++yi) {
                        double dt = ts - sae.at<double>(yi, xi);
                        if (dt < 0.01) {
                            add = true;
                            break;
                        }
                    }
                }
                if(!add) continue;
            }

            //we manually implement a filter here
            if(ts < p_sae + filter_thresh)
                continue;

            //update the sae
            p_sae = ts;

            //decay the valid region of the eros
            roi_raw.x = v->x - half_kernel;
            roi_raw.y = v->y - half_kernel;
            eros(roi_raw & roi_full) *= odecay;

            //set the eros position to max
            p_eros = 1.0;

            ts_curr = ts;
        }
        

    }

    template <typename T>
    void update(const T &packet)
    {
        update<T::iterator>(packet.begin(), packet.end());
    }

    jDot query_franco(int x, int y, int dRoi = 20, int dNei = 2, jDot pv = {0, 0}, bool circle = false)
    {
        static double eros_valid = 1.0 - eros_d;
        jDot out = {0, 0};
        int n = 0;

        int xl = std::max(x - dRoi, dNei); 
        int xh = std::min(x + dRoi, sae.cols - dNei);
        int yl = std::max(y - dRoi, dNei);
        int yh = std::min(y + dRoi, sae.rows - dNei);
        for (int yi = yl; yi <= yh; yi++)
        {
            for (int xi = xl; xi <= xh; xi++)
            {

                //keep searching if not a new events
                auto &ts = sae.at<double>(yi, xi);
                if(ts < ts_prev)
                    continue;

                //search neighbouring events to calc: distance / time
                int nn = 0;
                double xdot = 0.0, ydot = 0.0;
                for (auto yj = yi - dNei; yj <= yi + dNei; yj++)
                {
                    for (auto xj = xi - dNei; xj <= xi + dNei; xj++)
                    {
                        // if (eros.at<double>(yj, xi) >= eros_valid) {
                        if (eros.at<double>(yj, xj) >= eros_valid && sae.at<double>(yj, xj) != ts)
                        {
                            double inv_dt = 1.0 / (ts - sae.at<double>(yj, xj));
                            int dx = xi - xj;
                            int dy = yi - yj;
                            xdot += dx * inv_dt;
                            ydot += dy * inv_dt;
                            nn++;
                        }
                    }
                }
                // calculate the average for this event
                if (nn > 1)
                {
                    double inv_nn = 1.0 / nn;
                    out.u += xdot * inv_nn;
                    out.v += ydot * inv_nn;
                    n++;
                }
            }
        }

        if (n)
        {
            double inv_n = 1.0 / n;
            out.u *= inv_n;
            out.v *= inv_n;
        }

        //deal with inversion

        //add observation model



        return out;
    }

    // jDot query_fullroi(int x, int y, int d1 = 20, int d2 = 2, jDot pv = {0, 0})
    // {
    
    //     static double eros_valid = 1.0 - eros_d;
    //     jDot out = {0, 0};
    //     int nx = 0, ny = 0;

    //     int xl = std::max(x - d1, d2);
    //     int xh = std::min(x + d1, sae.cols - d2);
    //     int yl = std::max(y - d1, d2);
    //     int yh = std::min(x + d1, sae.rows - d2);
    //     for(int yi = yl; yi < yh; y++) {
    //         for(int xi = xl; xi < xh; x++) {

    //             //keep searching if not a new events
    //             auto &ts = sae.at<double>(yi, xi);
    //             if(ts <= ts_prev)
    //                 continue;

    //             //search neighbouring events to calc: distance / time
    //             int nnx = 0, nny = 0;
    //             double dtdx = 0.0, dtdy = 0.0;
    //             for(auto yj = yi - d2; yj < yi + d2; jy++) {
    //                 for(auto xj = xi - d2; xj < xi + d2; xj++) {
    //                     if (eros.at<double>(yj, xi) >= eros_valid) 
    //                     {
    //                         double dt = ts - sae.at<double>(yj, xj);
    //                         int dx = xi - xj;
    //                         int dy = yi - yj;
    //                         if(dx) {dtdx += dt / dx; nnx++;}
    //                         if(dy) {dtdy += dy / dy; nny++;}
    //                     }
    //                 }
    //             }
    //             // calculate the average for this event
    //             if(nnx) {out.u += 
    //             if (nn > 1) {
    //                 double inv_nn = 1.0 / nn;
    //                 out.u += xdot * inv_nn;
    //                 out.v += ydot * inv_nn;
    //                 n++;
    //             }
    //         }
    //     }

    //     if (n) {
    //         double inv_n = 1.0 / n; 
    //         out.u *= inv_n;
    //         out.v *= inv_n;
    //     }

    //     //deal with inversion

    //     //add observation model


    //     ts_prev = ts_curr;

    //     return out;
    // }

    skeleton13_vel query(skeleton13 points, int dRoi = 20, int dNei = 2, skeleton13_vel pv = {0}, bool circle = false)
    {
        skeleton13_vel out = {0};
        for(size_t i = 0; i < points.size(); i++)
            out[i] = query_franco(points[i].u, points[i].v, dRoi, dNei, pv[i], circle);
	    // ts_prev = ts_curr;
        return out;
    }


    jDot query_block(int x, int y, double evTs, int w , int h ,int dNei = 2)
    {

        // std::cout << x << ", " << y << std::endl;

        static double eros_valid = 1.0 - eros_d;
        jDot out = {0, 0};
        int n = 0;
        int nevs = 0;

        int xl = std::max(x - w, dNei); 
        int xh = std::min(x + w, sae.cols - dNei);
        int yl = std::max(y - h, dNei);
        int yh = std::min(y + h, sae.rows - dNei);

        // std::cout << x << " in [" << xl << ", " << xh << "]\t";
        // std::cout << y << " in [" << yl << ", " << yh << "]" << std::endl;

        for (int yi = yl; yi <= yh; yi++)
        {
            for (int xi = xl; xi <= xh; xi++)
            {
                nevs++;
                //keep searching if not a new events
                auto &ts = sae.at<double>(yi, xi);
                if(ts < ts_prev)
                    continue;

                //search neighbouring events to calc: distance / time
                int nn = 0;
                double xdot = 0.0, ydot = 0.0;
                for (auto yj = yi - dNei; yj <= yi + dNei; yj++)
                {
                    for (auto xj = xi - dNei; xj <= xi + dNei; xj++)
                    {
                        // if (eros.at<double>(yj, xi) >= eros_valid) {
                        if (eros.at<double>(yj, xj) >= eros_valid && sae.at<double>(yj, xj) != ts && sae.at<double>(yj, xj) > (evTs - 1e-1))
                        {
                            double inv_dt = 1.0 / (ts - sae.at<double>(yj, xj));
                            int dx = xi - xj;
                            int dy = yi - yj;
                            xdot += dx * inv_dt;
                            ydot += dy * inv_dt;
                            nn++;
                        }
                    }
                }
                // calculate the average for this event
                if (nn > 0)
                {
                    double inv_nn = 1.0 / nn;
                    out.u += xdot * inv_nn;
                    out.v += ydot * inv_nn;
                    n++;
                }
            }
        }

        if (n)
        {
            double inv_n = 1.0 / n;
            out.u *= inv_n;
            out.v *= inv_n;
        }

        // std::cout << nevs << std::endl;
        //deal with inversion

        //add observation model

        // std::cout << out.u << ", " << out.v << std::endl;

        return out;
    }

    void query_grid(std::vector<joint> &grid, double evTs, int rows, int cols, int dNei = 2)
    {
        int width = this->eros.cols/cols;
        int height = this->eros.rows/rows;
        // std::cout << width << " x " << height << std::endl;
        // std::vector<joint> gridV(rows*cols, {0.0, 0.0});
        // std::cout<< "gridEst\n";
        // for(size_t i = 0; i < gridV.size(); i++)
        //     gridV[i] = query_block(gridV[i].u, gridV[i].v, width, height, dNei);
        for(int i=0; i<rows; i++)
        {
            for(int j=0; j<cols; j++)
            {
                grid[i*cols+j].u = 0.0;
                grid[i*cols+j].v = 0.0;
                double cy = i*height+height/2;
                double cx = j*width+width/2;
                // std::cout << cx << ", " << cy << "    ";
                grid[i*cols+j] = query_block(cx, cy, evTs, width/2, height/2, dNei);
                // std::cout << grid[i*cols+j].u << ", " << grid[i*cols+j].v << "\t";
            }
            // std::cout << std::endl;
        }
        // std::cout << std::endl;
        // return gridV;
    }

    const cv::Mat& querySAE()
    {
        return sae;
    } 

    const cv::Mat& queryEROS()
    {
        return eros;
    }

};

class tripletVelocity
{
private:
    int roi_width{40};
    int minor_width{1};
    cv::Mat SAE, SAE_vis, SAE_out;
    cv::Mat SAEP, SAEN;
    // parameters for eros (used only for movenet)
    cv::Mat eros;
    cv::Rect roi_full;
    int eros_k{7};
    double eros_d{0.3};
    
public:
    void setParameters(int roi_width, int minor_width, cv::Size image_size) {
        this->roi_width = roi_width;
        this->minor_width = minor_width;
        this->SAEP = cv::Mat::zeros(image_size, CV_64F);
        this->SAEN = cv::Mat::zeros(image_size, CV_64F);
        // eros
        this->eros = cv::Mat(image_size, CV_64F);
        roi_full = cv::Rect(cv::Point(0, 0), image_size);
        this->eros_k = eros_k % 2 ? eros_k : eros_k + 1;
        this->eros_d = eros_d >= 1.0 ? 0.99 : eros_d;
        this->eros_d = eros_d < 0.0 ? 0.01 : eros_d;
    }

    template <typename T>
    skeleton13_vel update(const T &begin, const T &end, skeleton13 state, double evTs)
    {
        int n_used[13] = {0};
        skeleton13_vel velocity = {0};
        // eros
        //these are static so they get set once and we use the same memory
        //locations each call
        static double odecay = pow(eros_d, 1.0 / eros_k);
        static int half_kernel = eros_k / 2;
        static cv::Rect roi_raw = cv::Rect(0, 0, eros_k, eros_k);
        
        for(auto v_new = begin; v_new != end; v_new++) // each event
        {
            for (int k = 0; k < 13; k++) // each joint
            {
                // check if it's in the k-th roi and iff compute the velocity
                if (fabs(v_new->x - state[k].u) < roi_width/2 && fabs(v_new->y - state[k].v) < roi_width/2)
                {
                    int n_neighbours = 0;
                    double xdot = 0.0, ydot = 0.0;

                    // search for space-time neighbors for v_new= 1rs ev
                    for(int i = v_new->x-minor_width; i <= v_new->x+minor_width; i++) 
                    {
                        for(int j = v_new->y-minor_width; j <= v_new->y+minor_width; j++)
                        {
                            if((v_new->x!=i || v_new->y!=j)) // 2nd != 1st
                            {
                                // found the 2nd ev
                                double dt12;
                                if(v_new->p)
                                    dt12 = (v_new.timestamp() - this->SAEP.at<float>(j, i));
                                    // dt12 = (evTs - this->SAEP.at<float>(j, i));
                                else
                                    dt12 = (v_new.timestamp() - this->SAEN.at<float>(j, i));
                                    // dt12 = (evTs - this->SAEN.at<float>(j, i));
                                // search for 3rd ev (same polarity and similar dt)
                                for(int m = i-minor_width; m <= i+minor_width; m++)
                                {
                                    for(int n = j-minor_width; n <= j+minor_width; n++)
                                    {
                                        if((v_new->x!=m || v_new->y!=n) && (m!=i || n!=j)) //  3rd != 2nd != 1st
                                        {
                                            double dt23;
                                            if(v_new->p)
                                                dt23 = this->SAEP.at<float>(j, i) - this->SAEP.at<float>(n, m);
                                            else
                                                dt23 = this->SAEN.at<float>(j, i) - this->SAEN.at<float>(n, m);
                                            if(fabs(dt12-dt23) < fabs(dt12)/4)
                                            {
                                                int dx = v_new->x - i;
                                                int dy = v_new->y - j;
                                                xdot += dx / dt12;
                                                ydot += dy / dt12;
                                                n_neighbours++;
                                            }
                                        }
                                    }
                                }  
                            }
                        }
                    }
                    // update positve and negative SAE
                    // if(v_new->p)
                    //     this->SAEP.at<float>(v_new->y, v_new->x) = float(evTs);
                    // else   
                    //     this->SAEN.at<float>(v_new->y, v_new->x) = float(evTs);
                    // if(v_new->p)
                    //     this->SAEP.at<float>(v_new->y, v_new->x) = float(v_new.timestamp());
                    // else   
                    //     this->SAEN.at<float>(v_new->y, v_new->x) = float(v_new.timestamp());

                    // calculate the average for this event
                    if (n_neighbours > 0)
                    {
                        // std::cout << "N" << std::endl;
                        double inv_neighbours = 1.0 / n_neighbours; // (F) why doing this?
                        velocity[k].u += (xdot * inv_neighbours);
                        velocity[k].v += (ydot * inv_neighbours);
                        n_used[k]++;
                    }

                    
                }
            }
            // update positve and negative SAE
            if(v_new->p)
                this->SAEP.at<float>(v_new->y, v_new->x) = float(v_new.timestamp());
            else   
                this->SAEN.at<float>(v_new->y, v_new->x) = float(v_new.timestamp());
            // eros update
            double ts = v_new.timestamp();
            auto &p_eros = eros.at<double>(v_new->y, v_new->x);
            //decay the valid region of the eros
            roi_raw.x = v_new->x - half_kernel;
            roi_raw.y = v_new->y - half_kernel;
            eros(roi_raw & roi_full) *= odecay;

            //set the eros position to max
            p_eros = 1.0;            
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
      
        return velocity;
    }

    const cv::Mat& querySAEP()
    {
        // return SAEP;
        
        cv::normalize(this->SAEP, SAE_vis, 0, 1, cv::NORM_MINMAX);
        return SAE_vis;
        // cv::Mat SAEout;
        // SAE_vis.convertTo(SAE_out, CV_8U);
        // cv::cvtColor(this->SAEP, SAE_out, cv::COLOR_BGRA2GRAY);
        // return SAE_out;
    } 

    const cv::Mat& querySAEN()
    {
        cv::normalize(this->SAEN, SAE_vis, 0, 1, cv::NORM_MINMAX);
        return SAE_vis;
    }

    const cv::Mat& queryEROS()
    {
        return eros;
    } 
};

class pwTripletVelocity
{
private:
    int roi_width{40};
    int minor_width{1};
    cv::Mat SAE, SAE_vis, SAE_out;
    cv::Mat SAEP, SAEN;
    // parameters for eros (used only for movenet)
    cv::Mat eros;
    cv::Rect roi_full;
    int eros_k{7};
    double eros_d{0.3};

public:
    void setParameters(int roi_width, int minor_width, cv::Size image_size)
    {
        this->roi_width = roi_width;
        this->minor_width = minor_width;
        this->SAEP = cv::Mat::zeros(image_size, CV_32F);
        this->SAEN = cv::Mat::zeros(image_size, CV_32F);
        // eros
        this->eros = cv::Mat(image_size, CV_64F);
        roi_full = cv::Rect(cv::Point(0, 0), image_size);
        this->eros_k = eros_k % 2 ? eros_k : eros_k + 1;
        this->eros_d = eros_d >= 1.0 ? 0.99 : eros_d;
        this->eros_d = eros_d < 0.0 ? 0.01 : eros_d;
    }

    template <typename T>
    void updateSAE(const T &begin, const T &end, float evTs)
    {
        // eros
        //these are static so they get set once and we use the same memory
        //locations each call
        static double odecay = pow(eros_d, 1.0 / eros_k);
        static int half_kernel = eros_k / 2;
        static cv::Rect roi_raw = cv::Rect(0, 0, eros_k, eros_k);
        for(auto v_new = begin; v_new != end; v_new++) // each event
        {
            // float ts = v_new.timestamp();
            float ts = evTs;
            // std::cout << ts << std::endl;
            //update positve and negative SAEs
            if(v_new->p)
                this->SAEP.at<float>(v_new->y, v_new->x) = ts;
            else   
                this->SAEN.at<float>(v_new->y, v_new->x) = ts;

            // eros update
            auto &p_eros = eros.at<double>(v_new->y, v_new->x);
            //decay the valid region of the eros
            roi_raw.x = v_new->x - half_kernel;
            roi_raw.y = v_new->y - half_kernel;
            eros(roi_raw & roi_full) *= odecay;

            //set the eros position to max
            p_eros = 1.0;           
        }
    }

    jDot query_franco(int x, int y, float evTs, int dRoi = 20, int dNei = 2, jDot pv = {0, 0}, bool circle = false)
    {
        jDot out = {0, 0};
        int n = 0;

        int xl = std::max(x - dRoi, dNei); 
        int xh = std::min(x + dRoi, SAEP.cols - dNei);
        int yl = std::max(y - dRoi, dNei);
        int yh = std::min(y + dRoi, SAEP.rows - dNei);
        for (int yi = yl; yi <= yh; yi++)
        {
            for (int xi = xl; xi <= xh; xi++)
            {
                //keep searching if not a new events
                float &tsP = this->SAEP.at<float>(yi, xi);
                float &tsN = this->SAEN.at<float>(yi, xi);
                if(tsP < evTs && tsN < evTs)
                    continue;

                //search neighbouring events to calc: distance / time
                int nn = 0;
                double xdot = 0.0, ydot = 0.0;
                for (auto yj = yi - dNei; yj <= yi + dNei; yj++)
                {
                    for (auto xj = xi - dNei; xj <= xi + dNei; xj++)
                    {
                        if((xi!=xj || yi!=yj)) // 2nd != 1st
                        {
                            // found the 2nd ev
                            float dt12;
                            if(tsP >= evTs) // ev1 = postive event
                                dt12 = (evTs - this->SAEP.at<float>(yj, xj));
                            else if(tsN >= evTs)           // ev1 = negative event
                                dt12 = (evTs - this->SAEN.at<float>(yj, xj));

                            // search for 3rd ev (same polarity and similar dt)
                            for (auto yk = yj - dNei; yk <= yj + dNei; yk++)
                            {
                                for (auto xk = xj - dNei; xk <= xj + dNei; xk++)
                                {
                                    if((xi!=xk || yi!=yk) && (xk!=xj || yk!=yj)) //  3rd != 2nd != 1st
                                    {
                                        float dt23;
                                        if(tsP >= evTs) // ev1 = postive event
                                            dt23 = this->SAEP.at<float>(yj, xj) - this->SAEP.at<float>(yk, xk);
                                        else if(tsN >= evTs)           // ev1 = negative event
                                            dt23 = this->SAEN.at<float>(yj, xj) - this->SAEP.at<float>(yk, xk);
                                        // std::cout << dt12 << "\t" << dt23 << std::endl;
                                        if(fabs(dt12-dt23) < fabs(dt12)/2)
                                        {
                                            int dx = xi - xj;
                                            int dy = yi - yj;
                                            xdot += dx / dt12;
                                            ydot += dy / dt12;
                                            nn++;
                                        }    

                                    }       
                                }
                            }

                        }
                    }
                }
                // calculate the average for this event
                if (nn > 0)
                {
                    double inv_nn = 1.0 / nn;
                    out.u += xdot * inv_nn;
                    out.v += ydot * inv_nn;
                    n++;
                }
            }
        }

        if (n)
        {
            double inv_n = 1.0 / n;
            out.u *= inv_n;
            out.v *= inv_n;
        }

        return out;
    }

    skeleton13_vel query(skeleton13 points, double evTs, int dRoi = 20, int dNei = 2, skeleton13_vel pv = {0}, bool circle = false)
    {
        skeleton13_vel out = {0};
        for(size_t i = 0; i < points.size(); i++)
            out[i] = query_franco(points[i].u, points[i].v, (float) evTs, dRoi, dNei, pv[i], circle);
	    // ts_prev = ts_curr;
        return out;
    }

    const cv::Mat& queryEROS()
    {
        return eros;
    } 

    const cv::Mat& querySAEP()
    {
        return SAEP;
        // cv::Mat SAE_vis, SAEout;
        // cv::normalize(this->SAEP, SAE_vis, 0, 1, cv::NORM_MINMAX);
        // SAE_vis.convertTo(SAEout, CV_8U, 255);
        // return SAE_vis;
    } 

    const cv::Mat& querySAEN()
    {
        return SAEN;
        // cv::Mat SAE_vis, SAEout;
        // cv::normalize(this->SAEN, SAE_vis, 0, 1, cv::NORM_MINMAX);
        // SAE_vis.convertTo(SAEout, CV_8U, 255);
        // return SAE_vis;
    } 
};
}
