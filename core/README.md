# Core Library Functions for human pose estimation tasks

C++ src for compiling a library of functions useful for HPE:

* human pose estimation from events
* joint tracking
* skeleton tracking
* fusion
* on-line visualisation

# Build the library

  * Clone the repository: e.g. `git clone https://github.com/event-driven-robotics/hpe-core.git`
  * `cd hpe-core/core`
  * `mkdir build && cd build`
  * `cmake ..`
  * `make`

Cmake will search for installed dependencies, which are needed for the pose detection wrappers.

# Link the library to your repository

Using cmake, add the following to your 
  * 


