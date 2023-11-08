#pragma once
typedef unsigned short d_node_t;
enum class LaunchPolicy {SYNC, ASYNC};
enum ForcefieldType {WIRZ, PEDERSEN, FLATNESS_ENABLED};

#define SYCLBATCH
#include "isomer_batch.h"
template <int MaxDegree, typename T, typename K>
void dualise_sycl_v0        (sycl::queue& Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);

template <int MaxDegree, typename T, typename K>
void dualise_sycl_v1        (sycl::queue&Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);

template <typename T, typename K>
void tutte_layout           (sycl::queue&Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);

template <typename T, typename K>
void spherical_projection   (sycl::queue&Q, IsomerBatch<T,K>& B, const LaunchPolicy policy = LaunchPolicy::SYNC);

template <ForcefieldType FFT = PEDERSEN, typename T, typename K>
void forcefield_optimise    (sycl::queue&Q, IsomerBatch<T,K>& B, const int iterations, const int max_iterations, const LaunchPolicy policy = LaunchPolicy::SYNC);
