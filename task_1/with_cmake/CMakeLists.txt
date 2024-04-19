cmake_minimum_required(VERSION 3.0)
project(task1)

if(USE_FLOAT)
    add_definitions(-DTYPE_FLOAT)
endif()

if(USE_DOUBLE)
    add_definitions(-DTYPE_DOUBLE)
endif()

add_executable(task1 lab1.cpp)

