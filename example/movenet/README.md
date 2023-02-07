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
    $ cd hpe-core/example/movenet
    $ docker build -t movenet - < Dockerfile
    ```
:bulb: `<workspace>` is the parent directory of choice in which the repository is cloned
This will create a Docker image names movenet. Before running docker, instruct the host to accept GUI windows with the following command:
    
```shell
    $ xhost local:docker
```

Then run a container with the right parameters:

```shell
    $ docker run -it --privileged --network host -v /dev/bus/usb:/dev/bus/usb \
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

In case you wish to load any data present on host system, load its path when creating the container but adding another parameters in the command in the the format: `-v /path/on/host:/usr/local/data`

## Usage
To reopen a running container: 
```shell
    $ docker exec -it <container_name> bash
  ```

To run the movenet app, run 
```shell
    $ yarpmanager
  ```

and load the file `/usr/local/hpe-core/examples/movenet/yarpmanagerapplication.xml`. Run all and connect all to detect pose from the camera input

To create csv files from a stored dataset of eros frames:

```shell
  $ python3 evaluate.py --write_output --eval_img_path <<location_of_eros_frames>> \ 
  --eval_label_path <<location_to_json_file>> \
  --results_path <<location_to_save_csv_folder>>


```
