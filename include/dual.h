#pragma once
typedef unsigned short node_t;
#include "launch_ctx.h"
#include "cu_array.h"
#include <vector>
template <int MaxDegree> 
void dualise_V0(const node_t* G_in, const uint8_t* degrees, node_t* G_out , const int Nf,  const int N_graphs);

template <int MaxDegree>
void dualise_V0(const CuArray<node_t>& G_in, const CuArray<uint8_t>& degrees, CuArray<node_t>& G_out, const int Nf, const int N_graphs, const LaunchCtx& ctx = LaunchCtx(), const LaunchPolicy policy = LaunchPolicy::SYNC);

template <int MaxDegree>
void dualise_V1(const node_t* G_in, const uint8_t* degrees, node_t* G_out , const int Nf,  const int N_graphs);

template <int MaxDegree>
void dualise_V1(const CuArray<node_t>& G_in, const CuArray<uint8_t>& degrees, CuArray<node_t>& G_out, const int Nf, const int N_graphs, const LaunchCtx& ctx = LaunchCtx(), const LaunchPolicy policy = LaunchPolicy::SYNC);

template <int MaxDegree>
void dualise_V2(const std::vector<node_t>& G_in, const std::vector<uint8_t>& degrees, std::vector<node_t>& G_out , const int Nf,  const int N_graphs);

template <int MaxDegree>
void dualise_V3(const std::vector<node_t>& G_in, const std::vector<uint8_t>& degrees, std::vector<node_t>& G_out , const int Nf,  const int N_graphs);