#include "fusion.h"

using namespace hpecore;

bool stateEstimator::initialise(std::vector<double> parameters)
{
    return true;
}
void stateEstimator::updateFromVelocity(jointName name, jDot velocity, double dt)
{
    state[name] += (velocity * dt);
}

void stateEstimator::updateFromPosition(jointName name, joint position, double dt)
{
    state[name] = position;
}

void stateEstimator::updateFromVelocity(skeleton13 velocity, double dt)
{
    for (int j = 0; j < 13; j++)
        state[j] += (velocity[j] * dt);
}

void stateEstimator::updateFromPosition(skeleton13 position, double dt)
{
    state = position;
}

void stateEstimator::set(skeleton13 pose)
{
    state = pose;
    pose_initialised = true;
}

bool stateEstimator::poseIsInitialised()
{
    return pose_initialised;
}

skeleton13 stateEstimator::query()
{
    return state;
}

joint stateEstimator::query(jointName name)
{
    return state[name];
}

void kfEstimator::setTimePeriod(double dt)
{
    for (auto &kf : kf_array)
    {
        kf.processNoiseCov.at<float>(0, 0) = procU * dt;
        kf.processNoiseCov.at<float>(1, 1) = procU * dt;
    }
}

bool kfEstimator::initialise(std::vector<double> parameters)
{
    if (parameters.size() != 2)
        return false;
    procU = parameters[0];
    measU = parameters[1];

    // init(state size, measurement size)
    for (auto &kf : kf_array)
    {
        kf.init(2, 2);
        kf.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 0,
                               0, 1);

        kf.measurementMatrix = (cv::Mat_<float>(2, 2) << 1, 0,
                                0, 1);

        kf.processNoiseCov = (cv::Mat_<float>(2, 2) << procU, 0,
                              0, procU);
        kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) << measU, 0,
                                  0, measU);
    }
    return true;
}

void kfEstimator::updateFromVelocity(jointName name, jDot velocity, double dt)
{
    auto &kf = kf_array[name];
    joint j_new = state[name] + velocity * dt;
    setTimePeriod(dt);
    kf.predict();
    kf.correct((cv::Mat_<float>(2, 1) << j_new.u, j_new.v));
    state[name] = {kf.statePost.at<float>(0), kf.statePost.at<float>(1)};
}

void kfEstimator::updateFromPosition(jointName name, joint position, double dt)
{
    auto &kf = kf_array[name];
    setTimePeriod(dt);
    kf.predict();
    kf.correct((cv::Mat_<float>(2, 1) << position.u, position.v));
    state[name] = {kf.statePost.at<float>(0), kf.statePost.at<float>(1)};
}

void kfEstimator::updateFromPosition(skeleton13 position, double dt)
{
    for (auto &name : jointNames)
        updateFromPosition(name, position[name], dt);
}

void kfEstimator::set(skeleton13 pose)
{
    state = pose;
    for (jointName name : jointNames)
    {
        auto &kf = kf_array[name];
        kf.statePost.at<float>(0) = state[name].u;
        kf.statePost.at<float>(1) = state[name].v;
    }
    pose_initialised = true;
}

bool kfEstimator::poseIsInitialised()
{
    return pose_initialised;
}

skeleton13 kfEstimator::query()
{
    return state;
}

joint kfEstimator::query(jointName name)
{
    return state[name];
}