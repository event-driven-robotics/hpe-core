# Movenet.Pytorch

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/Fire/blob/main/LICENSE)

## Intro

MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body.
This is A Pytorch implementation of MoveNet inspired from the [Movenet.Pytorch](https://github.com/fire717/movenet.pytorch) modified to detect 13 keypoints trained on EROS, an event based representation, only consisting of the inference code.


## How To Run
1. Compile the Docker file to create the environment.


- Download the repository and build the Docker image
    ```shell
    $ cd <workspace>
    $ git clone git@github.com:event-driven-robotics/hpe-core.git
    $ cd hpe-core/example/yarp_glhpe
    $ docker build -t movenet - < Dockerfile
    ```
:bulb: `<workspace>` is the parent directory in which the repository is cloned

## Usage
- Run the Docker container and, inside it, run the gl-hpe pose detector
    ```shell
    $ docker run -it movenet bash
    ```

2.To test the model, run 
```
python evaluate.py
```

For run this setup on a new data, add the sample frames to folder data/ and run predict.py
```
python predict.py
```

## Resource
1. [Blog:Next-Generation Pose Detection with MoveNet and TensorFlow.js](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
)
2. [model card](https://storage.googleapis.com/movenet/MoveNet.SinglePose%20Model%20Card.pdf)
3. [TFHub：movenet/singlepose/lightning
](https://tfhub.dev/google/movenet/singlepose/lightning/4
)
4. [My article share: 2021轻量级人体姿态估计模型修炼之路（附谷歌MoveNet复现经验）](https://zhuanlan.zhihu.com/p/413313925
