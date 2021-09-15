
# GL-HPE Example
Online setup of the [Lifting events to 3D HPE](https://github.com/IIT-PAVIS/lifting_events_to_3d_hpe) program with a YARP input setup.
<!-- Use the hpecore installed with OpenPose functionality to perform HPE on greyscale images streamed over YARP. -->

The application has been designed to run using docker for simple set-up of the environment.

## Installation
The software was tested on Ubuntu 20.04.2 LTS without GPU support.

- Download the repository and build the Docker image
    ```shell
    $ cd <workspace>
    $ git clone git@github.com:event-driven-robotics/hpe-core.git
    $ cd hpe-core/example/yarp_glhpe
    $ docker build -t gl_hpe:yarp --ssh default --build-arg ssh_pub_key="$(cat ~/.ssh/<publicKeyFile>.pub)" --build-arg ssh_prv_key="$(cat ~/.ssh/<privateKeyFile>)" - < Dockerfile
    ```
:bulb: `<workspace>` is the parent directory in which the repository is cloned

:bulb: The ssh keys are required to access hpe-core as it is currently private. [Create a new ssh key](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) if required.

:warning: Ensure your ssh key is built without a passphrase.

## Usage
- Run the Docker container and, inside it, run the gl-hpe pose detector
    ```shell
    $ xhost +
    $ docker run -it -v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY gl_hpe:yarp bash
    ```
  
- At the terminal inside the container run the following commands
  ```shell 
  $ yarpserver &
  $ yarpmanager
  ```
  :warning: the `&` runs the process in the background enabling a single terminal to run both processes.

- In the `yarpmanager`, load the application from `/usr/local/code/hpe-core/example/yarp-glhpe` and run all modules

- In the `yarpdataplayer` GUI use the drop-down menus to load the test dataset at `/usr/local/code/hpe-core/example/test_dataset` and play

- Connect all connections in the `yarpmanager` 
  
- A new window should pop up and display the detected skeleton overlaid on the event frames