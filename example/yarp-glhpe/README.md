
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
    $ docker build -t gl_hpe:yarp < Dockerfile
    ```
:bulb: `<workspace>` is the parent directory in which the repository is cloned

## Usage
- Download the pre-trained model from [here](https://drive.google.com/drive/folders/1AgsQl6sTJBygPvgbdR1e9IfVAYxupMGI) and store it into folder `/path/to/pre/trained/model/folder`
- Run the Docker container
    ```shell
    $ xhost +
    $ docker run -it -v /tmp/.X11-unix/:/tmp/.X11-unix -v /path/to/pre/trained/model/folder:/usr/local/code/gl_hpe/checkpoint/ -e DISPLAY=unix$DISPLAY gl_hpe:yarp bash
    ```
  
- At the terminal inside the container run the following commands
  ```shell 
  $ yarpserver &
  $ yarpmanager
  ```
  :warning: the `&` runs the process in the background enabling a single terminal to run both processes.

- In the `yarpmanager`, load the application by opening `/usr/local/code/hpe-core/example/yarp-glhpe/app_yarp-glhpe.xml` 
  and run all modules (the first time the python script `yarp-glhpe/run-model.py` runs, it downloads an additional pre-trained 
  model for the backbone CNN which, according to the connection speed, might take several minutes and prevent pose prediction
  until it has finished)

- In the `yarpdataplayer` GUI use the drop-down menus to load the test dataset by opening folder `/usr/local/code/hpe-core/example/test_dataset` and play it

- Connect all connections in the `yarpmanager` 
  
- A new window will pop up and display the detected skeleton overlaid on the event frames
