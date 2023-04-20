#pragma once
typedef unsigned short d_node_t;
#ifdef CUDA_ENABLED
#include "launch_ctx.h"
#include "cu_array.h"
#endif
#include <vector>
#include <stdint.h>

//TODO: add versions which don't use CuArray, LaunchCtx, LaunchPolicy, but instead allocates memory itself, copies data to GPU, runs kernel, copies data back, and frees memory
template <int MaxDegree>
void dualise_V0(const d_node_t* G_in, const uint8_t* degrees, d_node_t* G_out , const int Nf,  const int N_graphs);

template <int MaxDegree>
void dualise_V1(const d_node_t* G_in, const uint8_t* degrees, d_node_t* G_out , const int Nf,  const int N_graphs);

#ifdef CUDA_ENABLED
template <int MaxDegree>
void dualise_V0(const CuArray<d_node_t>& G_in, const CuArray<uint8_t>& degrees, CuArray<d_node_t>& G_out, const int Nf, const int N_graphs, const LaunchCtx& ctx = LaunchCtx(), const LaunchPolicy policy = LaunchPolicy::SYNC);

template <int MaxDegree>
void dualise_V1(const CuArray<d_node_t>& G_in, const CuArray<uint8_t>& degrees, CuArray<d_node_t>& G_out, const int Nf, const int N_graphs, const LaunchCtx& ctx = LaunchCtx(), const LaunchPolicy policy = LaunchPolicy::SYNC);
#endif

template <int MaxDegree>
void dualise_V2(const std::vector<d_node_t>& G_in, const std::vector<uint8_t>& degrees, std::vector<d_node_t>& G_out , const int Nf,  const int N_graphs);

template <int MaxDegree>
void dualise_V3(const std::vector<d_node_t>& G_in, const std::vector<uint8_t>& degrees, std::vector<d_node_t>& G_out , const int Nf,  const int N_graphs);
