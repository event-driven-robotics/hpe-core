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

Note: You can also, instead, install the libraries named in `requirements.py` directly on your system.
## Usage
- Run the Docker container and, inside it, run the gl-hpe pose detector
    ```shell
    $ docker run -it movenet bash
    ```

2.To test the model, ensure you are in the examples/movenet folder and run 
```
python evaluate.py
```

For run this setup on a new data, add the sample frames to folder data/ and run predict.py
```
python predict.py
```
