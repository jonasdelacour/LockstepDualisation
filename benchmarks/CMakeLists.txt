add_executable(baseline baseline.cc)
target_link_libraries(baseline cpp_lib fullerenes)

add_executable(omp_multicore omp_multicore.cc)
target_link_libraries(omp_multicore cpp_lib fullerenes)

if(CMAKE_CUDA_COMPILER)
    add_executable(single_gpu single_gpu.cc)
    add_executable(multi_gpu multi_gpu.cc)
    target_link_libraries(single_gpu cuda_lib cpp_lib fullerenes)
    target_link_libraries(multi_gpu cuda_lib cpp_lib fullerenes)
endif()