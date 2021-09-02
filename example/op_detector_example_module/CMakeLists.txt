# requires minimum cmake version
cmake_minimum_required(VERSION 3.16.0)

# produce the cmake var PROJECT_NAME
project(op_detector_example_module)

#include(GNUInstallDirs)
# if(NOT CMAKE_CONFIGURATION_TYPES)
#     if(NOT CMAKE_BUILD_TYPE)
#         message(STATUS "Setting build type to 'Release' as none was specified.")
#         set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
#     endif()
# endif()

# find_package(YCM REQUIRED)
# include(AddInstallRPATHSupport)
# add_install_rpath_support(BIN_DIRS "${CMAKE_INSTALL_FULL_BINDIR}"
#                           LIB_DIRS "${CMAKE_INSTALL_FULL_LIBDIR}"
#                           INSTALL_NAME_DIR "${CMAKE_INSTALL_FULL_LIBDIR}"
#                           USE_LINK_PATH)



find_package(YARP COMPONENTS os sig REQUIRED)
# find_package(event-driven REQUIRED)
#find_package(OpenCV REQUIRED)
find_package(hpe-core REQUIRED)
# find_package(OpenPose REQUIRED)

#default the install location to that of event-driven

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

target_link_libraries(${PROJECT_NAME} YARP::YARP_OS 
                                      YARP::YARP_sig 
                                      hpe-core
                                      openpose
                                      ${OpenCV_LIBRARIES})

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})

#uninstall target?
