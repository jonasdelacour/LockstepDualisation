add_executable(baseline baseline.cc)
target_link_libraries(baseline cpp_lib fullerenes)

add_executable(omp_multicore omp_multicore.cc)
target_link_libraries(omp_multicore cpp_lib)

if(CMAKE_CUDA_COMPILER)
    add_executable(single_gpu single_gpu.cc)
    add_executable(multi_gpu multi_gpu.cc)
    target_link_libraries(single_gpu cpp_lib cuda_lib)
    target_link_libraries(multi_gpu cpp_lib cuda_lib)
endif()