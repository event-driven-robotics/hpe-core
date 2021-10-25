# requires minimum cmake version
cmake_minimum_required(VERSION 3.5)

# produce the cmake var PROJECT_NAME
project(glhpe-framer)

#find_package(YCM REQUIRED)
find_package(YARP COMPONENTS OS sig REQUIRED)
find_package(event-driven REQUIRED)

#default the install location to that of event-driven

add_executable(${PROJECT_NAME} glhpe-framer.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE YARP::YARP_os
                                              YARP::YARP_init
                                              ev::event-driven)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})

