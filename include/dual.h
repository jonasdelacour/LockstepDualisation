/* #pragma once
typedef unsigned short d_node_t;
enum class LaunchPolicy {SYNC, ASYNC};
enum ForcefieldType {WIRZ, PEDERSEN, FLATNESS_ENABLED};
#ifdef CUDA_ENABLED
#include "launch_ctx.h"
#include "cu_array.h"
#endif
#include <vector>
#include <stdint.h>

//TODO: add versions which don't use CuArray, LaunchCtx, LaunchPolicy, but instead allocates memory itself, copies data to GPU, runs kernel, copies data back, and frees memory

#ifdef CUDA_ENABLED
#define CUDABATCH
#include "isomer_batch.h"
template <int MaxDegree, typename T, typename K>
void dualise_cuda_v0(IsomerBatch<T,K>& B, const LaunchCtx& ctx = LaunchCtx(),       const LaunchPolicy policy = LaunchPolicy::SYNC);

template <int MaxDegree, typename T, typename K>
void dualise_cuda_v1(IsomerBatch<T,K>& B, const LaunchCtx& ctx = LaunchCtx(),       const LaunchPolicy policy = LaunchPolicy::SYNC);

template <typename T, typename K>
void tutte_layout(IsomerBatch<T,K>& B, const LaunchCtx& ctx = LaunchCtx(),          const LaunchPolicy policy = LaunchPolicy::SYNC);

template <typename T, typename K>
void spherical_projection(IsomerBatch<T,K>& B, const LaunchCtx& ctx = LaunchCtx(),  const LaunchPolicy policy = LaunchPolicy::SYNC);

template <ForcefieldType FFT = PEDERSEN, typename T, typename K>
void forcefield_optimise(IsomerBatch<T,K>& B, const int iterations, const int max_iterations, const LaunchCtx& ctx = LaunchCtx(), const LaunchPolicy policy = LaunchPolicy::SYNC);

#ifdef SYCL_ENABLED
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

#else
#define CPPBATCH
#include "isomer_batch.h"
template <int MaxDegree, typename T, typename K>
void dualise_omp_shared (IsomerBatch<T,K>& B);

template <int MaxDegree, typename T, typename K>
void dualise_omp_task   (IsomerBatch<T,K>& B);
#endif
 */