# requires minimum cmake version
cmake_minimum_required(VERSION 3.16.0)

# produce the cmake var PROJECT_NAME
project(python_cpp_test)

find_package(YARP COMPONENTS os sig REQUIRED)
find_package(hpe-core REQUIRED)
find_package(event-driven REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

target_link_libraries(${PROJECT_NAME} YARP::YARP_OS 
                                      YARP::YARP_sig 
                                      hpe-core::hpe-core)
                                      
add_executable(e2vid_framer e2vid_framer.cpp)

target_link_libraries(e2vid_framer YARP::YARP_OS 
                                   YARP::YARP_sig
                                   ev::event-driven 
                                   hpe-core::hpe-core)

install(TARGETS ${PROJECT_NAME} e2vid_framer DESTINATION ${CMAKE_INSTALL_BINDIR})

target_link_libraries(python_cpp_test YARP::YARP_OS
                                      Python3::Python)
