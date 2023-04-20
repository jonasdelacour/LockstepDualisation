add_executable(omp_validation omp_validation.cc)
target_link_libraries(omp_validation cpp_lib fullerenes)

if(CMAKE_CUDA_COMPILER)
    add_executable(gpu_validation gpu_validation.cc)
    target_link_libraries(gpu_validation cpp_lib cuda_lib fullerenes)
endif()
