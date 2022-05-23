# requires minimum cmake version
cmake_minimum_required(VERSION 3.5)

# produce the cmake var PROJECT_NAME
project(eros-framer)

#find_package(YCM REQUIRED)
find_package(YARP COMPONENTS os sig REQUIRED)
find_package(event-driven REQUIRED)
find_package(hpe-core REQUIRED)

#default the install location to that of event-driven

add_executable(${PROJECT_NAME} eros-framer.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE YARP::YARP_os
                                              YARP::YARP_init
                                              ev::event-driven
                                              hpe-core::hpe-core)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})

