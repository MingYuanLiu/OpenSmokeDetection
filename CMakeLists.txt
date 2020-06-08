cmake_minimum_required(VERSION 3.10)

project(smokeAdaboost CXX)

# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGES} -O -Wall -g --debug -std=c++11 -pthread")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -std=c++11 -pthread")

find_package(OpenCV REQUIRED)
list(APPEND CMAKE_INCLUDE_PATH /usr/local/include)
include_directories(${CMAKE_INCLUDE_PATH})

aux_source_directory(${PROJECT_SOURCE_DIR}/src/core DIR_LIB_SRCS)
add_executable(smokeAdaboost ${DIR_LIB_SRCS})
target_link_libraries(smokeAdaboost LINK_PRIVATE ${OpenCV_LIBS})

