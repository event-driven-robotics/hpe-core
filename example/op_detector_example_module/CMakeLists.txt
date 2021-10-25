# requires minimum cmake version
cmake_minimum_required(VERSION 3.16.0)

# produce the cmake var PROJECT_NAME
project(python_cpp_test)

find_package(YARP COMPONENTS os REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_executable(${PROJECT_NAME} python_cpp_test.cpp e2vid.cpp e2vid.h)

target_link_libraries(python_cpp_test YARP::YARP_OS
                                      Python3::Python)
