# requires minimum cmake version
cmake_minimum_required(VERSION 3.5)

# produce the cmake var PROJECT_NAME
project(volume_compensation)
find_package(YARP)
find_package(event-driven)
find_package(hpe-core)

#default the install location to that of event-driven
add_executable(${PROJECT_NAME} example_volumecomp.cpp visualisation.h)

target_link_libraries(${PROJECT_NAME} PRIVATE YARP::YARP_os
                                              YARP::YARP_init
					      hpe-core::hpe-core
                                              ev::event-driven)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})

