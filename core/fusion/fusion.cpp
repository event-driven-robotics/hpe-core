#include "fusion.h"

using namespace hpecore;

bool stateEstimator::initialise(std::vector<double> parameters)
{
    return true;
}
void stateEstimator::updateFromVelocity(jointName name, jDot velocity, double ts)
{
    double dt = ts - prev_ts;
    prev_ts = ts;
    state[name] += (velocity * dt);
}

void stateEstimator::updateFromPosition(jointName name, joint position, double ts)
{
    prev_ts = ts;
    state[name] = position;
}

void stateEstimator::updateFromVelocity(skeleton13 velocity, double ts)
{
    double dt = ts - prev_ts;
    prev_ts = ts;
    for (int j = 0; j < 13; j++)
        state[j] += (velocity[j] * dt);
}

void stateEstimator::updateFromPosition(skeleton13 position, double ts)
{
    prev_ts = ts;
    skeleton13_b valid = jointTest(position);
    for (auto &name : jointNames)
    {
        if(valid[name] && position[name].u && position[name].v)
        {
            state[name] = position[name];
        }
    }
    // state = position;
}

void stateEstimator::set(skeleton13 pose, double ts)
{
    prev_ts = ts;
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

skeleton13_vel stateEstimator::queryVelocity()
{
    return velocity;
}

void stateEstimator::setVelocity(skeleton13_vel vel)
{
    velocity = vel;
}

std::deque<hpecore::jDot>* stateEstimator::queryError()
{
    return error;
}

// ========================================================================== //

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
    if (parameters.size() != 3)
        return false;
    procU = parameters[0];
    measUD = parameters[1];
    measUV = parameters[2];
 
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
        kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) << measUD, 0,
                                  0, measUD);
    }
    return true;
}

void kfEstimator::updateFromVelocity(jointName name, jDot velocity, double ts)
{
    double dt = ts - prev_ts;
    prev_ts = ts;

    auto &kf = kf_array[name];
    joint j_new = state[name] + velocity * dt;
    setTimePeriod(dt);
    kf.measurementNoiseCov.at<float>(0, 0) = measUV;
    kf.measurementNoiseCov.at<float>(1, 1) = measUV;
    kf.predict();
    kf.correct((cv::Mat_<float>(2, 1) << j_new.u, j_new.v));
    state[name] = {kf.statePost.at<float>(0), kf.statePost.at<float>(1)};
}

void kfEstimator::updateFromPosition(jointName name, joint position, double ts)
{
    double dt = ts - prev_ts;
    prev_ts = ts;

    auto &kf = kf_array[name];
    setTimePeriod(dt);
    kf.measurementNoiseCov.at<float>(0, 0) = measUD;
    kf.measurementNoiseCov.at<float>(1, 1) = measUD;
    kf.predict();
    kf.correct((cv::Mat_<float>(2, 1) << position.u, position.v));
    state[name] = {kf.statePost.at<float>(0), kf.statePost.at<float>(1)};
}

void kfEstimator::updateFromPosition(skeleton13 position, double ts)
{
    skeleton13_b valid = jointTest(position);
    for (auto &name : jointNames)
        if(valid[name] && position[name].u && position[name].v)
            updateFromPosition(name, position[name], ts);
}

void kfEstimator::updateFromVelocity(skeleton13 velocity, double ts)
{
    for (auto &name : jointNames)
        updateFromVelocity(name, velocity[name], ts);
}

void kfEstimator::set(skeleton13 pose, double ts)
{
    prev_ts = ts;
    state = pose;
    for (jointName name : jointNames)
    {
        auto &kf = kf_array[name];
        kf.statePost.at<float>(0) = state[name].u;
        kf.statePost.at<float>(1) = state[name].v;
    }
    pose_initialised = true;
}

// ========================================================================== //

void constVelKalman::setTimePeriod(double dt)
{
    for (auto &kf : kf_array)
    {
        kf.processNoiseCov.at<float>(0, 0) = procU * dt;
        kf.processNoiseCov.at<float>(1, 1) = procU * dt;
        kf.processNoiseCov.at<float>(2, 2) = procUd * dt;
        kf.processNoiseCov.at<float>(3, 3) = procUd * dt;
        kf.transitionMatrix.at<float>(0, 2) = dt;
        kf.transitionMatrix.at<float>(1, 3) = dt;
    }
}

bool constVelKalman::initialise(std::vector<double> parameters)
{
    if (parameters.size() != 4)
        return false;
    procU = parameters[0];
    measU = parameters[1];
    procUd = parameters[2];
    measUd = parameters[3];

    measurement_position = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0);
    measurement_velocity = (cv::Mat_<float>(4, 4) << 0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1);

    // init(state size, measurement size)
    for (auto &kf : kf_array) {
        kf.init(4, 4);
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                               0, 1, 0, 1,
                               0, 0, 1, 0,
                               0, 0, 0, 1);

        kf.measurementMatrix = measurement_position;

        kf.processNoiseCov = (cv::Mat_<float>(4, 4) << procU, 0, 0, 0,
                              0, procU, 0, 0,
                              0, 0, procUd, 0,
                              0, 0, 0, procUd);

        kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) << measU, 0, 0, 0,
                                  0, measU, 0, 0,
                                  0, 0, measUd, 0,
                                  0, 0, 0, measUd);
    }

    return true;
}

void constVelKalman::updateFromVelocity(jointName name, jDot velocity, double ts)
{
    double dt = ts - prev_ts;
    prev_ts = ts;

    auto &kf = kf_array[name];
    setTimePeriod(dt);
    kf.measurementMatrix = measurement_velocity;
    kf.predict();
    kf.correct((cv::Mat_<float>(4, 1) << 0, 0, velocity.u, velocity.v));
    state[name] = {kf.statePost.at<float>(0), kf.statePost.at<float>(1)};
}

void constVelKalman::updateFromPosition(jointName name, joint position, double ts)
{
    double dt = ts - prev_ts;
    prev_ts = ts;

    auto &kf = kf_array[name];
    setTimePeriod(dt);
    kf.measurementMatrix = measurement_position;
    kf.predict();
    kf.correct((cv::Mat_<float>(4, 1) << position.u, position.v, 0, 0));
    state[name] = {kf.statePost.at<float>(0), kf.statePost.at<float>(1)};
}

void constVelKalman::updateFromPosition(skeleton13 position, double ts)
{
    for (auto &name : jointNames)
        updateFromPosition(name, position[name], ts);
}

void constVelKalman::updateFromVelocity(skeleton13 velocity, double ts)
{
    for (auto &name : jointNames)
        updateFromVelocity(name, velocity[name], ts);
}

void constVelKalman::set(skeleton13 pose, double ts)
{
    prev_ts = ts;
    state = pose;
    for (jointName name : jointNames)
    {
        auto &kf = kf_array[name];
        kf.statePost.at<float>(0) = state[name].u;
        kf.statePost.at<float>(1) = state[name].v;
        kf.statePost.at<float>(2) = 0.0f;
        kf.statePost.at<float>(3) = 0.0f;
    }
    pose_initialised = true;
}

// ========================================================================== //

void singleJointLatComp::updateFromPosition(joint position, double ts)
{
    // perform an (asynchronous) update of the position from the previous
    // position estimated (including the previous period velocity accumulation)
    double dt = ts - prev_pts;
    kf.processNoiseCov.at<float>(0, 0) = procU * dt;
    kf.processNoiseCov.at<float>(1, 1) = procU * dt;
    kf.predict();
    kf.correct((cv::Mat_<float>(2, 1) << position.u, position.v));

    // add the current period velocity accumulation to the state
    //vel_accum.u *= 5; vel_accum.v *= 5;
    std::cout << vel_accum.u << " " << vel_accum.v << std::endl;
    std::cout.flush();
    kf.statePost.at<float>(0) += vel_accum.u;
    kf.statePost.at<float>(1) += vel_accum.v;
    vel_accum = {0.0, 0.0};
}