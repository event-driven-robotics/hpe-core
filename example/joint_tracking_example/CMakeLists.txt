# requires minimum cmake version
cmake_minimum_required(VERSION 3.5)

# produce the cmake var PROJECT_NAME
project(joint_track)

#find_package(YCM REQUIRED)
find_package(YARP COMPONENTS OS sig cv REQUIRED)
find_package(hpe-core REQUIRED)
find_package(event-driven REQUIRED)

add_executable(${PROJECT_NAME}  joint_tracking_example.cpp
                                roiq.cpp
                                roiq.h)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
target_link_libraries(${PROJECT_NAME}   hpe-core::hpe-core
                                        YARP::YARP_os
                                        YARP::YARP_init
                                        YARP::YARP_cv
                                        ev::event-driven)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})

