# LockstepDualisation Artefact Description
This repository contains the code for the paper "Lockstep-Parallel Dualization of Surface Triangulations" by Jonas Dornonville de la Cour, Carl-Johannes Johnsen, and James Emil Avery.
Submitted to the 2023 International Conference for High Performance Computing, Networking, Storage, and Analysis.

# Instructions
## Software Prerequisites
* Linux or MacOS X (Tested on Ubuntu 18.04, Ubuntu 22.04, Arch Linux 6, MacOS X 13.3.1). CPU version works on Windows WSL on Ubuntu 22.04, GPU version does not due to Windows not supporting NVIDIA Unified Memory.
* CMake 3.18 or higher (Tested on Cmake 3.23 and 3.26)
* C++ compiler with C++17 support (Tested and verified with `gcc` 7.5, 11.3, and 12.2. Does not work with `clang`)
* Nvidia CUDA Toolkit 11.8 or higher
* Nvidia GPU with compute capability 5.0 or higher
* Git
* Fortran compiler (Tested with `gfortran` 7.5, 11.3, and 12.2)

## Build
### Quickstart

To automatically build, run automatic validation, and run benchmarks, simply type
```
make all
```
Each of the benchmarks produces a CSV file containing the results, and generates the benchmark plots. 
The benchmark and validation output will be placed in a directory named `output/<hostname>`.

To only build, or run benchmarks or validation separately, run
```
make build
```
or
```
make validation
```
or
```
make benchmarks
```
The validation checks the results from all the parallel implementations against the reference sequential dualization implementation in the Fullerene software package.
For every $n$ in $[20,24,26,\ldots,200]$, the check is performed against a random sample of 10,000 dual $C_n$ fullerene isomer graphs (or the full isomer space if smaller than 10,000).
We verify that the results are identical.

The benchmarks can also be performed interactively with the Jupyter notebook, `reproduce.ipynb`.

### Manual build
In case the automatic build fails for some reason, the individual steps to build and run the software is as follows:

1. Fetch the Fullerene software package as a submodule (for reference comparisons)
```
git submodule update --init
```

2. After this, the benchmarks can be built using `CMake` and `make`:
```
mkdir build
cd build/
cmake ..
make -j
```

## Manual Run
After building, return to the repository root directory before running the benchmarks.
The executables are located in the `build/benchmarks` and `build/validation` directories. The executables are:
```
build/benchmarks/baseline
build/benchmarks/omp_multicore
build/benchmarks/single_gpu
build/benchmarks/multi_gpu
```
The GPU benchmarks will only be built if the CUDA toolkit is available.

All executables take the same command line parameters. For example:
```
./build/benchmarks/multi_gpu <Ntriangles> <Ngraphs> <Nruns> <Nwarmup> <variant:0|1>
```
1. `Ntriangles`: one of [20, 24, 26, 28, ... , 200] (to match the fullerene test-data). Default: 200
2. `Ngraphs` : batch size, i.e. the number of graphs to dualise in parallel. 
3. `Nruns`: number of repeated runs. Default: 10. To reproduce results from the paper, set to 100 (but takes longer).
4. 'Nwarmup`: number of warmup runs. Default: 1
5. `variant`: Kernel variant.
  - For GPU, kernel 0 uses one thread per triangle (`Ntriangles` threads), and kernel 1 uses one thread per vertex.
  - For CPU, kernel 0 is the shared-memory parallel version, and kernel 1 is the task-parallel version.

For example,
```
./build/benchmarks/single_gpu 100 1000000 100 1 1
```
runs the single-GPU benchmark for a million C100 fullerene isomers, repeated 100 times for statistics, with a single warmup-run, using GPU kernel 1.


