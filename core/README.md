# Core Library Functions for human pose estimation tasks

C++ src for compiling a library of functions useful for HPE:

* human pose estimation from events
* joint tracking
* skeleton tracking
* position and velocity fusion
* on-line visualisation

# Build the library

The library is designed to use CMake to configure and build.

* Clone the repository: e.g. `git clone https://github.com/event-driven-robotics/hpe-core.git`
* `cd hpe-core/core`
* `mkdir build && cd build`
* `cmake ..`
* `make`

CMake will search for installed dependencies, which are needed for the pose detection wrappers. To install specific dependencies, check the Dockerfile in example modules for tips. If you want all dependencies we provide a (large) Dockerfile [here](todo).

# Link the library to your repository

Using cmake, add the following to your `CMakeLists.txt`

*  `find_package(hpe-core)`
*  `target_link_libraries(${PROJECT_NAME} PRIVATE hpe-core::hpe-core)`

# Example

Examples of how to install dependencies, build the library, and link it in your own project can be found [here](https://github.com/event-driven-robotics/hpe-core/tree/main/example/op_detector_example_module). The Dockerfile can be used to automatically build the dependencies required in a isolated container, or if you prefer, your can follow it's instructions to install the dependencies natively on your machine. [Interested to learn more about Docker?](https://www.docker.com/)
