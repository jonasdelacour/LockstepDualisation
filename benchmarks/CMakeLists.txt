add_executable(single_gpu single_gpu.cc)
add_executable(multi_gpu multi_gpu.cc)
target_link_libraries(single_gpu cuda_lib)
target_link_libraries(multi_gpu cuda_lib)