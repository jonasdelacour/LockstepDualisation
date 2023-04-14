#Set Up CMake

cmake_minimum_required(VERSION 3.1)

if (${APPLE}) # Mac OS X
    # Change to use brew gcc rather than clang
    set(CMAKE_C_COMPILER "gcc-12")
    set(CMAKE_CXX_COMPILER "g++-12")
endif()

project(Dualisation)

set(CMAKE_CXX_STANDARD 17)
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_FLAGS "-std=c++1z -O3 -fPIC -m64 -pthread -fopenmp")

include_directories(AFTER SYSTEM include)
include_directories(${CMAKE_BINARY_DIR}/include/)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-I/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/11.8/include/")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-std=c++17 --use_fast_math --fmad true --ftz true --maxrregcount=40 -Xptxas -v --resource-usage -gencode arch=compute_86,code=sm_86 -I/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/11.8/include/ -I./include/")
    add_subdirectory(src/cuda)
endif()

add_subdirectory(src/c++)
add_subdirectory(benchmarks)