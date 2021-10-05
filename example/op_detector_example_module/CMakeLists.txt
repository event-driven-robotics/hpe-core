# requires minimum cmake version
cmake_minimum_required(VERSION 3.16.0)

# produce the cmake var PROJECT_NAME
project(op_detector_example_module)

find_package(YARP COMPONENTS os sig REQUIRED)
find_package(hpe-core REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(OpenCV)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

target_link_libraries(${PROJECT_NAME} YARP::YARP_OS 
                                      YARP::YARP_sig 
                                      hpe-core::hpe-core)

add_executable(python_cpp_test python_cpp_test.cpp)

target_link_libraries(python_cpp_test ${OpenCV_LIBRARIES} Python3::Python Python3::NumPy)  #PUBLIC??

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})

# define C++ macro specifying e2vid path that will be used at runtime
add_compile_definitions(E2VID_PYTHON_DIR=$ENV{E2VID_PYTHON_DIR})