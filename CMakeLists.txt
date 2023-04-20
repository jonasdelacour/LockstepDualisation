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

set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -fPIC -m64 -pthread -fopenmp")

include_directories(AFTER SYSTEM include)
include_directories(include)
include_directories(external/fullerenes/include)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CUDA_INCLUDE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  
  message("CUDA include path at: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
  message("CUDA ARCHITECTURES: ${CUDA_ARCHITECTURES}")
  
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -I${CUDA_INCLUDE} -DCUDA_ENABLED")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -std=c++17 --use_fast_math --fmad true --ftz true --maxrregcount=40 -Xptxas -v --resource-usage -arch native -I${CUDA_INCLUDE} -DCUDA_ENABLED")
    add_subdirectory(src/cuda)
endif()

add_subdirectory(src/c++)
add_subdirectory(external/fullerenes EXCLUDE_FROM_ALL)
add_subdirectory(benchmarks)
add_subdirectory(validation)


