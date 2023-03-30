#Set Up CMake

cmake_minimum_required(VERSION 3.1)
project(Dualisation)

set(CMAKE_CXX_STANDARD 17)
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_FLAGS "-std=c++1z -O3 -fPIC -m64 -pthread -fopenmp -I/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/11.8/include/")

add_subdirectory(benchmarks)

include_directories(AFTER SYSTEM include)
include_directories(${CMAKE_BINARY_DIR}/include/)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-std=c++17 --use_fast_math --fmad true --ftz true --maxrregcount=40 -Xptxas -v --resource-usage -gencode arch=compute_86,code=sm_86 -I/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/11.8/include/ -I./include/")
    add_subdirectory(src)
endif()