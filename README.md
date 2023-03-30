# LockstepDualisation
Standalone repository for lockstep parallel dualisation. This repository contains the code for the paper "Lockstep-Parallel Cubic Graph-dualization" by Jonas Dornonville de la Cour, Carl-Johannes Johnsen, and James Emil Avery. Submitted to the 2023 International Conference for High Performance
Computing, Networking, Storage, and Analysis

## Instructions
### Prerequisites
* CMake 3.1 or higher
* C++ compiler with C++17 support
* Nvidia CUDA Toolkit 11.7 or higher
* Nvidia GPU with compute capability 7.0 or higher

### Build
```mkdir build && cd build```

```cmake ..```

```make```

### Run
```./benchmarks/single_gpu```

```./benchmarks/multi_gpu```

Arguments can be passed to the executables:
1. Number of vertices in each graph. Valid range: [20, 24, 26, 28, ... , 200]
2. Number of graphs to dualise. Maximum value depends on available GPU memory.
3. Number of runs to perform. Default value: 100
4. Kernel variant to use default value: 0, Valid values: [0, 1]
