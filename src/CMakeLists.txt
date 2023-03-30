enable_language(CXX CUDA)

add_library(cuda_lib SHARED cu_array.cu launch_ctx.cu dual.cu)
add_library(cpp_lib SHARED dual_omp.cc)
