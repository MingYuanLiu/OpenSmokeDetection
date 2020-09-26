if(CMAKE_TOOLCHAIN_FILE)
    # get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
    get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
    find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
    message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 3.10)

project(smokeAdaboost CXX)

option(DRAW_RESULTS "draw results" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -std=c++11 -pthread")

if (DRAW_RESULTS)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DDRAW_RESULTS")
endif()

find_package(OpenCV REQUIRED)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/core DIR_LIB_SRCS)
add_executable(smokeAdaboost ${DIR_LIB_SRCS})
target_link_libraries(smokeAdaboost LINK_PRIVATE ${OpenCV_LIBS})

# add_subdirectory(lib)
set(EXCLUDE_MAIN_FILE "${PROJECT_SOURCE_DIR}/src/core/main.cpp")
list(REMOVE_ITEM DIR_LIB_SRCS ${EXCLUDE_MAIN_FILE})
add_library(smoke_adaboost SHARED ${DIR_LIB_SRCS})
add_library(smoke_adaboost_static STATIC ${DIR_LIB_SRCS})
SET_TARGET_PROPERTIES(smoke_adaboost_static PROPERTIES OUTPUT_NAME "smoke_adaboost")
SET_TARGET_PROPERTIES(smoke_adaboost_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
SET_TARGET_PROPERTIES(smoke_adaboost PROPERTIES CLEAN_DIRECT_OUTPUT 1)
SET_TARGET_PROPERTIES(smoke_adaboost PROPERTIES VERSION 1.0 SOVERSION 1)

file(GLOB _PROJECT_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/src/core/*.hpp)
INSTALL(TARGETS smoke_adaboost smoke_adaboost_static LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
INSTALL(FILES ${_PROJECT_INCLUDE_PATH} DESTINATION include/smoke_adaboost)


