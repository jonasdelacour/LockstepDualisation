#Set Up CMake
project(LockstepDualisation)

cmake_minimum_required(VERSION 3.5)
include(CheckLanguage)
include(CMakeDependentOption)

set(CMAKE_GENERATOR "Unix Makefiles" CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Make external fullerene package use a Fortran compiler that works.
set(CMAKE_Fortran_COMPILER gfortran CACHE STRING "Fortran compiler") # Default to GNU Fortran (e.g. Cray Fortran breaks currently)
check_language(fortran)

if (${APPLE}) # Mac OS X
    # Change to use brew gcc rather than clang
    set(CMAKE_C_COMPILER "gcc-12")
    set(CMAKE_CXX_COMPILER "g++-12")
endif()

#--------- CROSS-ARCHITECTURE GPU HANDLING -----
set(GPU_MAXREGCOUNT 80 CACHE STRING "Control GPU register pressure vs. occupancy")

# Do we have CUDA compilers?
check_language(CUDA)
# Do we want to build CUDA?
cmake_dependent_option(ENABLE_CUDA "Build CUDA GPU code" ON "CMAKE_CUDA_COMPILER" OFF)
if(CMAKE_CUDA_COMPILER)
  message("Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE INTEGER "Architecture for your NVIDIA GPU: 50|60|70|75|80|86|89|90. See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/")
else()
  message("Could not find CUDA compiler.")
endif()

# Can we do a SYCL? Find icpx or SYCL-enabled clang.
find_program(SYCL_LS sycl-ls)
if(SYCL_LS)
  get_filename_component(SYCL_BIN_DIR ${SYCL_LS} DIRECTORY)
  set(SYCL_DIR ${SYCL_BIN_DIR}/../)
  find_program(DPCPP_COMPILER NAMES ${SYCL_BIN_DIR}/icpx ${SYCL_BIN_DIR}/clang++)
  
  if(DPCPP_COMPILER)
    message("Found dpc++ compiler: ${DPCPP_COMPILER}")
  else()
    message("Could not find dpc++ compiler")
  endif()
endif()
# Do we want to build SYCL?
cmake_dependent_option(ENABLE_SYCL "Build SYCL GPU code" ON "DPCPP_COMPILER" OFF)    

# If we want to build SYCL, we need to set up some command line options
# that depend on the target architecture.
if(ENABLE_SYCL)
  if(CMAKE_CUDA_COMPILER)
    set(SYCL_DEFAULT_TARGET "NVIDIA")
  else()
    set(SYCL_DEFAULT_TARGET "AMD")
  endif()
  
  set(SYCL_TARGETS ${SYCL_DEFAULT_TARGET} CACHE STRING "AMD|NVIDIA|X86 (Currently only single target supported.") # TODO: Multi-target compiles.
  message("SYCL compile enabled with ${SYCL_TARGETS} target")

  set(DPCPP_FLAGS "-O3 -fsycl -fsycl-unnamed-lambda") # Use regardless of target.
  
  if(SYCL_TARGETS STREQUAL "AMD") 
    set(DPCPP_TARGET_FLAGS "-fsycl-targets=amdgcn-amd-amdhsa,x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -march=native")
    set(DPCPP_LINK_FLAGS   "-O3 -fsycl -fsycl-unnamed-lambda ${DPCPP_TARGET_FLAGS}")
    
  elseif(SYCL_TARGETS STREQUAL "NVIDIA")
    set(PTXAS_COMPILE_FLAGS    " --verbose -g --maxrregcount=${GPU_MAXREGCOUNT} --opt-level=3 --generate-line-info --warn-on-double-precision-use --warn-on-spills")
    set(PTXAS_LINK_FLAGS
    "SHELL:-Xcuda-ptxas --verbose"
    "SHELL:-Xcuda-ptxas --maxrregcount=80"
    "SHELL:-Xcuda-ptxas --opt-level=3"
    "SHELL:-Xcuda-ptxas --generate-line-info"
    "SHELL:-Xcuda-ptxas --warn-on-double-precision-use"
    "SHELL:-Xcuda-ptxas --warn-on-spills"
    "SHELL:-Xcuda-ptxas --generate-line-info")
    
    set(DPCPP_TARGET_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda,x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_${CMAKE_CUDA_ARCHITECTURES}")
    set(DPCPP_FLAGS       "${DPCPP_FLAGS} -Xcuda-ptxas '${PTXAS_COMPILE_FLAGS}'")
    set(DPCPP_LINK_FLAGS   "-O3 -fsycl -fsycl-unnamed-lambda ${DPCPP_TARGET_FLAGS}")

    
  elseif(SYCL_TARGETS STREQUAL "X86")
    set(DPCPP_TARGET_FLAGS "-fsycl-targets=x86_64 -march=native")
    set(DPCPP_LINK_FLAGS   "-O3 -fsycl -fsycl-unnamed-lambda ${DPCPP_TARGET_FLAGS}")
    
  endif()

  set(DPCPP_INCLUDE ${SYCL_DIR}/include)  
  set(DPCPP_FLAGS "${DPCPP_FLAGS} -fsycl -fsycl-unnamed-lambda ${DPCPP_TARGET_FLAGS} -I${DPCPP_INCLUDE} -I${DPCPP_INCLUDE}/sycl")

endif()


# ----------------------------------------------

include_directories(AFTER SYSTEM include)
include_directories(AFTER SYSTEM ${CMAKE_BINARY_DIR}/include)
include_directories(external/fullerenes/include)

if(CMAKE_CUDA_COMPILER)
  message("CUDA version: ${CUDA_VERSION}")
  message("CUDA toolkit root directory: ${CUDA_TOOLKIT_ROOT_DIR}")
  message("CUDA devices: ${CUDA_DEVICES}")
  enable_language(CUDA)
  set(CUDA_INCLUDE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  
  message("CUDA include path at: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
  message("CUDA ARCHITECTURES: ${CUDA_ARCHITECTURES}")
  
  set(CUDA_INCLUDE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -I${CUDA_INCLUDE} -DCUDA_ENABLED")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -std=c++${CMAKE_CXX_STANDARD} --use_fast_math --fmad true --ftz true --maxrregcount=${GPU_MAXREGCOUNT} -Xptxas -v --resource-usage -arch native -I${CUDA_INCLUDE} -DCUDA_ENABLED")
    add_subdirectory(src/cuda)
endif()

add_subdirectory(src/c++)
add_subdirectory(external/fullerenes EXCLUDE_FROM_ALL)
add_subdirectory(benchmarks)
add_subdirectory(validation)

if(DPCPP_COMPILER)
  set(PREVIOUS_CXX_COMPILER ${CMAKE_CXX_COMPILER})
  set(CMAKE_CXX_COMPILER    ${DPCPP_COMPILER})
  add_subdirectory(src/sycl)
  add_subdirectory(validation/sycl)
  add_subdirectory(benchmarks/sycl)
  set(CMAKE_CXX_COMPILER   ${PREVIOUS_CXX_COMPILER})
endif()


# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Configuration with access to CMake results from code
configure_file(
  ${CMAKE_SOURCE_DIR}/include/config.hh.in
  ${CMAKE_BINARY_DIR}/include/config.hh
  )

