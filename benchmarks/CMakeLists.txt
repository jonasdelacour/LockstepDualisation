link_libraries(cpp_lib fullerenes)
add_executable(baseline baseline.cc)
add_executable(buckybench buckybench.cc)


if(OpenMP_FOUND)
  add_executable(omp_multicore omp_multicore.cc)
  target_link_libraries(omp_multicore PUBLIC cpp_lib  OpenMP::OpenMP_CXX)
else()
  message("OpenMP-enabled compiler not found.")
endif()

if(CMAKE_CUDA_COMPILER)
    add_executable(single_gpu single_gpu.cc)
    add_executable(multi_gpu multi_gpu.cc)

    target_link_libraries(single_gpu cuda_lib cpp_lib )
    target_link_libraries(multi_gpu cuda_lib cpp_lib  )
endif()

