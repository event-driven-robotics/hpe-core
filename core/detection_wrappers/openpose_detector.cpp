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
#include "openpose_detector.h"

using namespace hpecore;

bool OpenPoseDetector::init(std::string models_path, std::string pose_model, std::string size)
{
    try
    {
        // description of detector's parameters can be found at
        // https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
        const auto poseMode = op::flagsToPoseMode(1);  // body keypoints detection
        const auto poseModel = op::flagsToPoseModel(op::String(pose_model));  
        // 'BODY_25', 25 keypoints, fastest with CUDA, most accurate, includes foot keypoints
        // 'COCO', 18 keypoints
        // 'MPI', 15 keypoints, least accurate model but fastest on CPU
        // 'MPI_4_layers', 15 keypoints, even faster but less accurate

        // TODO: set poseJointsNum (get number of joints from openpose mapping)

        size = "-1x"+size;
        const auto netInputSize = op::flagsToPoint(op::String(size), size.c_str());
        const auto outputSize = op::flagsToPoint(op::String("-1x-1"), "-1x-1");
        const auto keypointScaleMode = op::flagsToScaleMode(0);
        const auto multipleView = false;
        const auto heatMapTypes = op::flagsToHeatMaps(false, false, false);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(2);
        const bool enableGoogleLogging = false;
        const auto num_gpu = -1;
        const auto num_gpu_start = 0;
        const auto scale_number = 1;
        const auto scale_gap = 0.25;
        const auto render_pose = -1;
        const auto disable_blending = false;
        const auto alpha_pose = 0.6;
        const auto alpha_heatmap = 0.7;
        const auto part_to_show = 0;
        const auto part_candidates = false;
        const auto render_threshold = 0.05;
        const auto number_people_max = -1;
        const auto maximize_positives = false;
        const auto fps_max = -1;
        const auto prototxt_path = "";
        const auto caffemodel_path = "";
        const auto upsampling_ratio = 0.;
        const auto disable_multi_thread = true;    /////////////////////////////////
        // process_real_time  ////////////////////////////////////////

        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, num_gpu,
            num_gpu_start, scale_number, (float)scale_gap,
            op::flagsToRenderMode(render_pose, multipleView), poseModel, !disable_blending,
            (float)alpha_pose, (float)alpha_heatmap, part_to_show, op::String(models_path),
            heatMapTypes, heatMapScaleMode, part_candidates, (float)render_threshold,
            number_people_max, maximize_positives, fps_max, op::String(prototxt_path),
            op::String(caffemodel_path), (float)upsampling_ratio, enableGoogleLogging};
        detector.configure(wrapperStructPose);
        

        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (disable_multi_thread)
            detector.disableMultiThreading();
        
        detector.start();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        // TODO: print error on stdout
        return false;
    }

    return true;
}

void OpenPoseDetector::stop()
{
    detector.stop();
}

skeleton13 OpenPoseDetector::detect(cv::Mat &input)
{
    skeleton18 pose = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    skeleton13 pose13 = {0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 };

    try
    {
        const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(input);
        auto datumProcessed = detector.emplaceAndPop(imageToProcess);
        if (datumProcessed == nullptr)
        {
            std::cout << "Image could not be processed" << std::endl;
            return pose13;
        }
        input = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData);
        //cv::imshow("inlib", temp);
        //cv::waitKey(1);


        // parse and return keypoints
        const auto& poseKeypoints = datumProcessed->at(0)->poseKeypoints;
        // for (auto person = 0; person < poseKeypoints.getSize(0); person++)
        // {



        // TODO: code assumes openpose returns only one pose, make it compatible with multiple poses
        for (auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
        {
            // get (x, y) coords

            pose[bodyPart] = {poseKeypoints[{0, bodyPart, 0}], poseKeypoints[{0, bodyPart, 1}]};
        }

        // }
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }

    pose13 = hpecore::coco18_to_dhp19(pose);
    return pose13;
}


void openposethread::run()
{
    while (true)
    {
        m.lock();
        if (stop)
            return;
        pose.pose = detop.detect(image);
        data_ready = true;
    }
}


bool openposethread::init(std::string model_path, std::string model_name, std::string model_size)
{
    // initialise open pose
    if (!detop.init(model_path, model_name, model_size))
        return false;

    // make sure the thread won't start until an image is provided
    m.lock();

    // make sure that providing an image will start things for the first go
    data_ready = true;

    // start the thread
    th = std::thread([this]
                        { this->run(); });

    return true;
}

void openposethread::close()
{
    stop = true;
    m.try_lock();
    m.unlock();
}

bool openposethread::update(cv::Mat next_image, double image_timestamp, hpecore::stampedPose &previous_result)
{
    // if no data is ready (still processing) do nothing
    if (!data_ready)
        return false;

    // else set the result to the provided stampedPose
    previous_result = pose;
    previous_result.delay = image_timestamp - pose.timestamp;

    // set the timestamp
    pose.timestamp = image_timestamp;

    // and the image for the next detection
    static cv::Mat img_u8, img_float;
    next_image.copyTo(img_float);
    double min_val, max_val;
    cv::minMaxLoc(img_float, &min_val, &max_val);
    max_val = std::max(fabs(max_val), fabs(min_val));
    img_float /= (2 * max_val);
    img_float += 0.5;
    img_float.convertTo(img_u8, CV_8U, 255, 0);
    cv::cvtColor(img_u8, image, cv::COLOR_GRAY2BGR);

    // and unlock the procesing thread
    m.try_lock();
    m.unlock();
    data_ready = false;
    return true;
}
