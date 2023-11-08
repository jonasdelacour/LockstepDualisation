if(ICPX_COMPILER)
    message(STATUS "Using ICPX compiler: ${CMAKE_CXX_COMPILER}")
    include_directories(/opt/intel/oneapi/compiler/2023.2.1/linux/include/sycl/CL)
    set_source_files_properties(sycl_validation.cc PROPERTIES COMPILE_FLAGS "${ICPX_FLAGS}")
    add_executable(sycl_validation sycl_validation.cc)
    target_include_directories(sycl_validation PRIVATE "/opt/intel/oneapi/compiler/2023.2.1/linux/include/sycl" "/opt/intel/oneapi/compiler/2023.2.1/linux/include")
    target_link_libraries(sycl_validation sycl_fullerene sycl fullerenes)
    target_link_options(sycl_validation PRIVATE 
        -O3
        -fsycl
        "SHELL:${ICPX_TARGET_FLAG}"
        "SHELL:-Xcuda-ptxas --maxrregcount=40"
    )
endif()