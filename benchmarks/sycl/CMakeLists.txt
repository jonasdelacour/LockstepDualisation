if(DPCPP_COMPILER)


list(APPEND SYCL_EXECUTABLES dualisation pipeline baseline_pipeline)
list(APPEND REGCOUNTS 40 80 80)

foreach(EXECUTABLE IN ZIP_LISTS SYCL_EXECUTABLES)
  set_source_files_properties(${EXECUTABLE_0}.cc PROPERTIES COMPILE_FLAGS "${DPCPP_FLAGS} ${DPCPP_TARGET_FLAGS}")    
endforeach()

foreach(EXECUTABLE IN ZIP_LISTS SYCL_EXECUTABLES REGCOUNTS)
  add_executable(${EXECUTABLE_0} ${EXECUTABLE_0}.cc)    
  target_include_directories(${EXECUTABLE_0} PRIVATE ${DPCPP_INCLUDE} ${DPCPP_INCLUDE}/sycl)
  target_link_libraries(${EXECUTABLE_0} sycl_fullerene sycl fullerenes cpp_lib)
  target_link_options(${EXECUTABLE_0} PRIVATE     
    "SHELL:${DPCPP_LINK_FLAGS}"
    "SHELL:-Xcuda-ptxas --maxrregcount=${EXECUTABLE_1}"
    ${PTXAS_LINK_FLAGS}
  )
endforeach()

endif()
