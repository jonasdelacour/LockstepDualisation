#pragma once
typedef unsigned short d_node_t;
enum class LaunchPolicy {SYNC, ASYNC};
enum ForcefieldType {WIRZ, PEDERSEN, FLATNESS_ENABLED};

#include "launch_ctx.h"
#include "cu_array.h"

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
