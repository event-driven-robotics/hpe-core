# requires minimum cmake version
cmake_minimum_required(VERSION 3.5)

# produce the cmake var PROJECT_NAME
project(reproj-test)

#find_package(YCM REQUIRED)
find_package(hpe-core REQUIRED)

#default the install location to that of event-driven

add_executable(${PROJECT_NAME} test.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE hpe-core::hpe-core)


