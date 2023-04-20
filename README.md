# LockstepDualisation
Standalone repository for lockstep parallel dualisation. This repository contains the code for the paper "Lockstep-Parallel Cubic Graph-dualization" by Jonas Dornonville de la Cour, Carl-Johannes Johnsen, and James Emil Avery. Submitted to the 2023 International Conference for High Performance
Computing, Networking, Storage, and Analysis

# Instructions
## Prerequisites
* CMake 3.1 or higher
* C++ compiler with C++17 support (Tested and verified with `gcc` 11.3 and 12.2. Does not work with `clang`)
* Nvidia CUDA Toolkit 11.7 or higher
* Nvidia GPU with compute capability 7.0 or higher
* Git
* Fortran compiler (Tested with `gfortran` 11.3 and 12.2)

## Build
In order to build the benchmarks, the submodules must be cloned, followed by applying a patch allowing for ARM compilation:
```
git submodule update --init
cd external/fullerenes
git apply ../../fullerenes.patch
```
After this, the benchmarks can be built using CMake:
```
mkdir build
cd build/
cmake ..
make -j
```

## Run
After building, the executables are located in the `build/benchmarks` directory. The executables are:
```
build/benchmarks/baseline
build/benchmarks/omp_multicore
```
If the benchmarks were built with CUDA support, the following executables are also available:
```
build/benchmarks/single_gpu
build/benchmarks/multi_gpu

Currently must be run from the root directory of the repository.
```

The following arguments can be passed to the executables:
1. Number of vertices in each graph. Valid range: [20, 24, 26, 28, ... , 200]
2. Number of graphs to dualise. Maximum value depends on available GPU memory.
3. Number of runs to perform. Default value: 100
4. Number of warmup runs to perform. Default value: 1
5. Kernel variant to use default value: 0, Valid values: [0, 1] (ignored for the `baseline` benchmark)

# TODO THIS SHOULD BE IMPLEMENTED AND THEN BE MERGED WITH THE RUN DESCRIPTION ABOVE:
Each of the benchmarks produces a CSV file containing the results. A script to run all of the benchmarks and generate the plots from the CSV files is also included in `plotting.ipynb`.