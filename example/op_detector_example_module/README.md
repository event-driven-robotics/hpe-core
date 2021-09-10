# OpenPose Example
Use the hpecore installed with OpenPose functionality to perform HPE on greyscale images streamed over YARP.

The application has been designed to run using docker for simple set-up of the environment.

## Installation
The software was tested on Ubuntu 20.04.2 LTS with an Nvidia GPU.

- Install the latest [Nvidia driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)
- Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu)
- Install [Nvidia Docker Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- Download the repository and build the Docker image
    ```shell
    $ cd /path/to/repository/folder
    $ docker build -t op-yarp --ssh default --build-arg ssh_pub_key="$(cat ~/.ssh/<publicKeyFile>)" --build-arg ssh_prv_key="$(cat ~/.ssh/<privateKeyFile>)" - < Dockerfile
    ```
Note: Ensure your ssh key is built without a passphrase.

## Usage
- Run the Docker container and, inside it, run the pose detector
    ```shell
    $ xhost +
    $ docker run -it -v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --runtime=nvidia -v </path_to_dataset>:/usr/data op-yarp
    ```
  
- At the terminal inside the container run the following commands
  ```shell 
  yarpserver &
  yarpdataplayer &
  op_detector_example_module &
  ```
  :warning: the `&` runs the process in the background enabling a single terminal to run all three processes.

- In the `yarpdataplayer` GUI select load an appropriate dataset

- Connect the output port of the `yarpdatplayer` to the input port of the `op_detector_example_module`
  ```shell 
  yarp connect /ATIS/img:o /op_detector_example_module/img:i fast_tcp
  ```

- Press play on the `yarpdataplayer` and the `op_detector_example_module` should display the detected skeleton overlaid on the images
