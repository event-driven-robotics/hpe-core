# requires minimum cmake version
cmake_minimum_required(VERSION 3.5)

# produce the cmake var PROJECT_NAME
project(joint_track)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-variable -I/usr/include/python3.8 -lpython3.8")

#find_package(YCM REQUIRED)
find_package(YARP COMPONENTS OS sig REQUIRED)
find_package(hpe-core REQUIRED)
find_package(event-driven REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Python3 COMPONENTS Development NumPy)

include_directories(${PYTHON_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

#default the install location to that of event-driven

add_executable(${PROJECT_NAME}  joint_tracking_example.cpp
                                roiq.cpp
                                roiq.h
                                matplotlibcpp.h)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
target_link_libraries(${PROJECT_NAME}   hpe-core::hpe-core
                                        YARP::YARP_os
                                        YARP::YARP_init
                                        ev::event-driven
                                        stdc++fs
                                        Python3::Python Python3::NumPy)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})

