# Movenet.Pytorch

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/Fire/blob/main/LICENSE)

## Intro

MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body.
This is A Pytorch implementation of a variation of MoveNet inspired from the [Movenet.Pytorch](https://github.com/fire717/movenet.pytorch), modified to detect 13 keypoints trained on EROS, an event based representation. This example only consists of the inference code.

![moveEnet](https://github.com/user-attachments/assets/9ea56f92-a22a-4202-8340-1a0a6faeec73)


## Installation

The environment can be installed in 2 different ways:

1. Compile the Docker file to create the environment.

- Download the repository and build the Docker image
    ```shell
    $ cd <workspace>
    $ git clone git@github.com:event-driven-robotics/hpe-core.git
    $ cd hpe-core/example/movenet
    $ docker build -t moveEnet - < Dockerfile
    ```
-`<workspace>` is the parent directory of choice in which the repository is cloned
- If your computer does not have a GPU, replace `Dockerfile` with `Dockerfile_cpu` 

his will create a Docker image names movenet. 

Before running docker, instruct the host to accept GUI windows with the following command:
    
```shell
    $ xhost local:docker
```

Then run a container with the right parameters:

```shell
    $ docker run -it --privileged --network host -v /dev/bus/usb:/dev/bus/usb 
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY \
    --name <container_name> movenet:latest
```

The meaning of the options are:
* -it : runs the container in an interactive bash
* --privileged : give the docker rights to open devices (to read the camera over usb)
* -v /dev/bus/usb:/dev/bus/usb : mount the /dev folder so the docker can find the right device
* -v /tmp/.X11-unix:/tmp/.X11-unix : mount the temporary configuration for x11 to use the same options as the host
* -e DISPLAY=unix$DISPLAY : set the environment variable for X11 forwarding
* --name : name your container so you remember it, if not specified docker will assign a random name that you will have to remember

In case you wish to load any data present on host system or save results to a location external to the docker container, load a path as a volume when creating the container by adding another parameters in the command in the format: `
-v /path/on/host:/usr/local/data`

2. Create a python environment on you local machine. 
If you want to run this offline only, this option can be used. 
 - Create a virtual environment and enter it.
 - Install dependencies from the requirements.txt:

```shell
  $ python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
```
 - Add hpe core to your PYTHONPATH
```shell
  $ export PYTHONPATH=$PYTHONPATH:/path/to/hpe-core/
```
## Usage (Inference)

To reopen a running container: 
```shell
    $ docker exec -it <container_name> bash
  ```

### Online Inference on Camera input
To run online pose estimation on a camera input, in commandline, run:  
```shell
    $ yarpmanager
  ```

then load the file `/usr/local/hpe-core/examples/movenet/yarpmanagerapplication.xml`. Run all and connect all to detect pose from the camera input

### Offline Inference on a data sample

A event sample from the [event-human 3.6m dataset](https://zenodo.org/records/7842598) is provided. To see the result on it, run: 

```shell
    $ mkdir data && cd data && wget https://github.com/user-attachments/files/17645984/cam2_S1_Directions.zip
    $ unzip cam2_S1_Directions.zip && cd ..
    $ python3 moveEnet-offline.py -visualise False -write_video data/cam2_S1_Directions/moveEnet.mp4 -input data/cam2_S1_Directions/ch0dvs/
```
Note: You can point the -write_video path to the host volume if you mounted one while creating the container.

