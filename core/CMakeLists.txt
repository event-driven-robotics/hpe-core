# Copyright: (C) 2021 EDPR - Istituto Italiano di Tecnologia
# Authors: Arren Glover

cmake_minimum_required(VERSION 3.16.0)

project(hpe-core
        LANGUAGES CXX
        VERSION 0.1)

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

#set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(MSVC)
    set(CMAKE_DEBUG_POSTFIX "d")
endif()

#set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build libraries as shared as opposed to static" ON)

find_package(YCM)
#set_package_properties(YCM PROPERTIES TYPE RECOMMENDED)

#if(NOT YCM_FOUND)
#  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#endif()

include(AddInstallRPATHSupport)
add_install_rpath_support(BIN_DIRS "${CMAKE_INSTALL_FULL_BINDIR}"
                          LIB_DIRS "${CMAKE_INSTALL_FULL_LIBDIR}"
                          INSTALL_NAME_DIR "${CMAKE_INSTALL_FULL_LIBDIR}"
                          USE_LINK_PATH)

if(NOT CMAKE_CONFIGURATION_TYPES)
    if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "Setting build type to 'Release' as none was specified.")
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
    endif()
endif()

find_package(OpenCV)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: (found version ${OpenCV_VERSION})")
endif()

set( folder_source
  detection_wrappers/detection.cpp
)

set( folder_header
  utility/utility.h
  detection_wrappers/detection.h
)

find_package(OpenPose)
if( OpenPose_FOUND )
  message(STATUS "Found Openpose: (found version ${OpenPose_VERSION})")
  list(APPEND folder_source detection_wrappers/openpose_detector.cpp)
  list(APPEND folder_header detection_wrappers/openpose_detector.h)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
if( Python3_FOUND )
  list(APPEND folder_source utility/e2vid/e2vid.cpp)
  list(APPEND folder_header utility/e2vid/e2vid.h)
endif()


add_library(${PROJECT_NAME} ${folder_source} ${folder_header})

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${folder_header}"
                                                          VERSION ${hpe-core_VERSION}
                                                          SOVERSION 2)                                               

target_include_directories(${PROJECT_NAME} PRIVATE utility
                                           PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>") # FIXME INSTALL PATH

target_compile_options(${PROJECT_NAME} PRIVATE -Wall)

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenPose_LIBS} ${OpenCV_LIBRARIES} Python3::Python Python3::NumPy)

install(TARGETS ${PROJECT_NAME}
        EXPORT hpecore
        LIBRARY       DESTINATION "${CMAKE_INSTALL_LIBDIR}"                     COMPONENT shlib
        ARCHIVE       DESTINATION "${CMAKE_INSTALL_LIBDIR}"                     COMPONENT lib
        RUNTIME       DESTINATION "${CMAKE_INSTALL_BINDIR}"                     COMPONENT bin
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}" COMPONENT dev)

include(InstallBasicPackageFiles)
install_basic_package_files(${PROJECT_NAME}
                            EXPORT hpecore
                            VERSION ${${PROJECT_NAME}_VERSION}
                            COMPATIBILITY SameMajorVersion
                            LOWERCASE_FILENAMES
                            NAMESPACE hpe-core::
                            DEPENDENCIES OpenCV)
include(AddUninstallTarget)

#########
# E2VID #
#########

# copy e2vid python files to hpe-core install dir
#add_custom_command(
#        TARGET ${PROJECT_NAME} POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy
#                ${CMAKE_SOURCE_DIR}/test/input.txt
#                ${CMAKE_CURRENT_BINARY_DIR}/input.txt)

# define C++ macro specifying e2vid path that will be used at runtime
add_compile_definitions(E2VID_PYTHON_DIR="$ENV{E2VID_PYTHON_DIR}")
