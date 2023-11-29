if(DPCPP_COMPILER)
    message(STATUS "Using DPCPP compiler: ${CMAKE_CXX_COMPILER}")
#    include_directories(/opt/intel/oneapi/compiler/2023.2.1/linux/include/sycl/CL) # No! Bad! 
    set_source_files_properties(sycl_benchmark.cc PROPERTIES COMPILE_FLAGS "${DPCPP_FLAGS}")
    add_executable(sycl_benchmark sycl_benchmark.cc)
#    target_include_directories(sycl_benchmark PRIVATE "/opt/intel/oneapi/compiler/2023.2.1/linux/include/sycl" "/opt/intel/oneapi/compiler/2023.2.1/linux/include")
    target_link_libraries(sycl_benchmark sycl_fullerene sycl fullerenes cpp_lib)
    target_link_options(sycl_benchmark PRIVATE 
        -O3
        -fsycl
        "SHELL:${DPCPP_TARGET_FLAG}"
        "SHELL:-Xcuda-ptxas --maxrregcount=40"
    )
endif()
