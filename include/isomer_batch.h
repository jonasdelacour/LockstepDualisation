#pragma once
#include <type_traits>
#include <array>
#include <stddef.h>
#define FLOAT_TYPEDEFS(T) static_assert(std::is_floating_point<T>::value, "T must be float"); typedef std::array<T,3> coord3d; typedef std::array<T,2> coord2d; typedef T real_t;
#define INT_TYPEDEFS(K) static_assert(std::is_integral<K>::value, "K must be integral type"); typedef std::array<K,3> node3; typedef std::array<K,2> node2; typedef K node_t; typedef std::array<K,6> node6;
#define TEMPLATE_TYPEDEFS(T,K) FLOAT_TYPEDEFS(T) INT_TYPEDEFS(K)

#if defined(SYCLBATCH)
    #include <CL/sycl.hpp>
    #define BUFFER(T) sycl::buffer<T, 1>
#elif defined(CPPBATCH)
    #include <vector>
    #define BUFFER(T) std::vector<T>
#elif defined(CUDABATCH)
    #include <cuda.h>
    #include <cuda_runtime.h>
    #define BUFFER(T) T*
#endif

enum class IsomerStatus
{
    EMPTY,
    CONVERGED,
    PLZ_CHECK,
    FAILED,
    NOT_CONVERGED
};

template <typename T, typename K>
struct IsomerBatch
{
    TEMPLATE_TYPEDEFS(T, K);
    size_t m_capacity = 0;
    size_t n_atoms = 0;
    size_t n_faces = 0;

    BUFFER(coord3d)         X;
    BUFFER(coord2d)         xys;
    BUFFER(K)               cubic_neighbours;
    BUFFER(K)               dual_neighbours;
    BUFFER(K)               face_degrees;
    BUFFER(size_t)          IDs;
    BUFFER(size_t)          iterations;
    BUFFER(IsomerStatus)    statuses;
    bool allocated = false;
    // std::vector<std::tuple<void**,size_t,bool>> pointers;
    IsomerBatch(size_t n_atoms, size_t n_isomers);
    IsomerBatch();
};

//Copy function
template <typename T, typename K>
void copy(IsomerBatch<T,K>& dest, IsomerBatch<T,K>& src);

