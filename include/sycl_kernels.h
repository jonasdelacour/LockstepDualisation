#pragma once
typedef unsigned short d_node_t;
enum class LaunchPolicy {SYNC, ASYNC};
enum ForcefieldType {WIRZ, PEDERSEN, FLATNESS_ENABLED, FLAT_BOND, BOND, ANGLE, DIH, ANGLE_M, ANGLE_P, DIH_A, DIH_M, DIH_P};

#define SYCLBATCH
#include "isomer_batch.h"
template <int MaxDegree, typename T, typename K>
void dualise_sycl_v1            (sycl::queue& Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);

template <int MaxDegree, typename T, typename K>
void dualise_sycl_v2            (sycl::queue&Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);

template <int MaxDegree, typename T, typename K>
void dualise_sycl_v3            (sycl::queue&Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);

template <int MaxDegree, typename T, typename K>
void dualise_sycl_v4            (sycl::queue&Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);

template <typename T, typename K>
void tutte_layout_sycl          (sycl::queue&Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);

template <typename T, typename K>
void spherical_projection_sycl  (sycl::queue&Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);

template <ForcefieldType FFT = PEDERSEN, typename T, typename K>
void forcefield_optimise_sycl   (sycl::queue&Q, IsomerBatch<T,K>& B, const int iterations, const int max_iterations, const LaunchPolicy policy = LaunchPolicy::SYNC);

//Use this kernel to trigger a memory transfer from the device to the host.
template <typename T, typename K>
void nop_kernel                 (sycl::queue&Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);
