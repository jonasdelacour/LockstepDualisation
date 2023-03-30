#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include "dual.h"
#include "cu_array.h"
#include "iostream"
typedef ushort2 edge_t;
namespace cg = cooperative_groups;

template<int MaxDegree>
struct DeviceDualGraph{
    const node_t* dual_neighbours;                          //(Nf x MaxDegree)
    const uint8_t* face_degrees;                            //(Nf x 1)
    
    __device__ DeviceDualGraph(const node_t* dual_neighbours, const uint8_t* face_degrees) : dual_neighbours(dual_neighbours), face_degrees(face_degrees) {}

    __inline__ __device__ node_t dedge_ix(const node_t u, const node_t v) const{
        for (uint8_t j = 0; j < face_degrees[u]; j++){
            if (dual_neighbours[u*MaxDegree + j] == v) return j;
        }

        assert(false);
	    return 0;		// Make compiler happy
    }

    /**
     * @brief returns the next node in the clockwise order around u
     * @param v the current node around u
     * @param u the node around which the search is performed
     * @return the next node in the clockwise order around u
     */
    __device__ node_t next(const node_t u, const node_t v) const{
        node_t j = dedge_ix(u,v);
        return dual_neighbours[u*MaxDegree + ((j+1)%face_degrees[u])];
    }
    
    /**
     * @brief returns the prev node in the clockwise order around u
     * @param v the current node around u
     * @param u the node around which the search is performed
     * @return the previous node in the clockwise order around u
     */
    __device__ node_t prev(const node_t u, const node_t v) const{
        node_t j = dedge_ix(u,v);
        return dual_neighbours[u*MaxDegree + ((j-1+face_degrees[u])%face_degrees[u])];
    }

    /**
     * @brief Find the node that comes next on the face. given by the edge (u,v)
     * @param u Source of the edge.
     * @param v Destination node.
     * @return The node that comes next on the face.
     */
    __device__ node_t next_on_face(const node_t u, const node_t v) const{
        return prev(v,u);
    }

    /**
     * @brief Find the node that comes next on the face. given by the edge (u,v)
     * @param u Source of the edge.
     * @param v Destination node.
     * @return The node that comes next on the face.
     */
    __device__ node_t prev_on_face(const node_t u, const node_t v) const{
        return next(v,u);
    }

    /**
     * @brief Finds the cannonical triangle arc of the triangle (u,v,w)
     * 
     * @param u source node
     * @param v target node
     * @return cannonical triangle arc 
     */
    __device__ edge_t get_cannonical_triangle_arc(const node_t u, const node_t v) const{
        //In a triangle u, v, w there are only 3 possible representative arcs, the cannonical arc is chosen as the one with the smalles source node.
        edge_t min_edge = {u,v};
        node_t w = next(u,v);
        if (v < u && v < w) min_edge = {v, w};
        if (w < u && w < v) min_edge = {w, u};
        return min_edge;
    }
};

/** @brief Computes the exclusive scan of the data elements provided by each thread 
 * 
 * @param sdata A pointer to shared memory.
 * @param data The data to be scanned.
 * @param n The number of elements in the scan.
*/
template <typename T> __device__ void ex_scan(T* sdata, const T data, int n){
    auto warpid = threadIdx.x >> 5;
    auto lane = threadIdx.x & 31;

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());

    auto result = cg::inclusive_scan(tile32, data);

    if (lane == 31){
        sdata[n+1 + warpid] = result;
    }

    __syncthreads();
    if (warpid == 0){
        auto val = cg::inclusive_scan(tile32, sdata[n+1 + lane]);
        sdata[n+1 + lane] = val;
    }
    __syncthreads();
    if (warpid == 0)
    {
        sdata[threadIdx.x + 1] = result;
    } else{
        if (threadIdx.x < n) {
        sdata[threadIdx.x + 1] =  sdata[n+1 + warpid-1] + result;}
        
    }
    if (threadIdx.x == 0){
        sdata[0] == (T)0;
    }
    __syncthreads();
}


template <int MaxDegree>
__global__
void dualise_V0_(const node_t* G_in, const uint8_t* deg, node_t* G_out , const int N_f,  const int N_graphs){
    auto N_t = 2*(N_f - 2);
    extern __shared__  node_t sharedmem[];
    node_t* triangle_numbers = reinterpret_cast<node_t*>(sharedmem);
    node_t* cached_neighbours = triangle_numbers + N_f*MaxDegree;
    uint8_t* cached_degrees = reinterpret_cast<uint8_t*>(cached_neighbours+ N_f*MaxDegree);
    node_t* smem = reinterpret_cast<node_t*>(cached_neighbours) + N_f*(MaxDegree + 1);
    reinterpret_cast<float*>(smem)[threadIdx.x] = 0;
    for (int isomer_idx = blockIdx.x; isomer_idx < N_graphs; isomer_idx += gridDim.x ){
    __syncthreads();
    auto thid = threadIdx.x;
    if (thid < N_f){
        memcpy(cached_neighbours + thid*MaxDegree, G_in + (thid + isomer_idx*N_f)*MaxDegree, sizeof(node_t)*MaxDegree);
        cached_degrees[thid] = deg[thid + isomer_idx*N_f];
    }
    DeviceDualGraph<MaxDegree> FD(cached_neighbours, cached_degrees);
    node_t cannon_arcs[MaxDegree]; memset(cannon_arcs, UINT16_MAX,sizeof(node_t)*MaxDegree);
    int represent_count = 0; //The number of triangles that the thid'th node is representative of.
    __syncthreads();
    //Step 1: Find canonical arcs of triangles and store the arcs locally (canon_arcs) and count number of triangles represented by thid.
    if(thid < N_f){
        for (int i = 0; i < FD.face_degrees[thid]; ++i){
            edge_t cannon_arc = FD.get_cannonical_triangle_arc(thid,FD.dual_neighbours[thid*MaxDegree + i]);
            if (cannon_arc.x == thid) {
                cannon_arcs[i] = cannon_arc.y;
                represent_count++;
            }
        }
    }
    //Step 2: Scan and store in lookup table
    ex_scan<node_t>(smem,represent_count,N_f);
    
    if (thid < N_f){
        int arc_count = 0;
        for (size_t i = 0; i < FD.face_degrees[thid]; ++i){
            if (cannon_arcs[i] != UINT16_MAX) {
                triangle_numbers[thid*MaxDegree + i] = smem[thid] + arc_count;
                ++arc_count;
            }
        }
    }

    __syncthreads(); //TODO check if this is needed
    //Step 3: Store the arcs representing each triangle in an ordered list (representative_arc_list)
    edge_t* representative_arc_list = reinterpret_cast<edge_t*>(smem);
    if (thid < N_f){
        for (size_t i = 0; i < FD.face_degrees[thid]; i++){
            if(cannon_arcs[i] != UINT16_MAX){
                auto idx = triangle_numbers[thid*MaxDegree + i];
                representative_arc_list[idx] = {node_t(thid), cannon_arcs[i]}; 
            }
        }
        
    }
    __syncthreads();
    //Step 4: Find neighbouring triangles through their canonical arcs and find their IDs by looking up in triangle_numbers
    auto [u, v] = representative_arc_list[thid];
    node_t w = FD.next(u,v);

    edge_t edge_b = FD.get_cannonical_triangle_arc(v, u); G_out[isomer_idx*N_t*3 + thid*3 + 0] = triangle_numbers[edge_b.x * MaxDegree + FD.dedge_ix(edge_b.x, edge_b.y)];
    edge_t edge_c = FD.get_cannonical_triangle_arc(w, v); G_out[isomer_idx*N_t*3 + thid*3 + 1] = triangle_numbers[edge_c.x * MaxDegree + FD.dedge_ix(edge_c.x, edge_c.y)];
    edge_t edge_d = FD.get_cannonical_triangle_arc(u, w); G_out[isomer_idx*N_t*3 + thid*3 + 2] = triangle_numbers[edge_d.x * MaxDegree + FD.dedge_ix(edge_d.x, edge_d.y)];
    }
}

template <int MaxDegree>
__global__
void dualise_V1_(const node_t* G_in, const uint8_t* deg, node_t* G_out , const int N_f,  const int N_graphs){
    auto N_t = 2*(N_f - 2);
    extern __shared__  node_t sharedmem[];
    node_t* triangle_numbers = reinterpret_cast<node_t*>(sharedmem);
    node_t* cached_neighbours = triangle_numbers + N_f*MaxDegree;
    uint8_t* cached_degrees = reinterpret_cast<uint8_t*>(cached_neighbours+ N_f*MaxDegree);
    node_t* smem = reinterpret_cast<node_t*>(cached_neighbours) + N_f*(MaxDegree + 1);
    reinterpret_cast<float*>(smem)[threadIdx.x] = 0;
    for (int isomer_idx = blockIdx.x; isomer_idx < N_graphs; isomer_idx += gridDim.x ){
    __syncthreads();
    auto thid = threadIdx.x;
    if (thid < N_f){
        memcpy(cached_neighbours + thid*MaxDegree, G_in + (thid + isomer_idx*N_f)*MaxDegree, sizeof(node_t)*MaxDegree);
        cached_degrees[thid] = deg[thid + isomer_idx*N_f];
    }
    DeviceDualGraph<MaxDegree> FD(cached_neighbours, cached_degrees);
    node_t cannon_arcs[MaxDegree]; memset(cannon_arcs, UINT16_MAX,sizeof(node_t)*MaxDegree);
    int represent_count = 0; //The number of triangles that the thid'th node is representative of.
    __syncthreads();
    //Step 1: Find canonical arcs of triangles and store the arcs locally (canon_arcs) and count number of triangles represented by thid.
    if(thid < N_f)
    for (int i = 0; i < FD.face_degrees[thid]; ++i){
        edge_t cannon_arc = FD.get_cannonical_triangle_arc(thid,FD.dual_neighbours[thid*MaxDegree + i]);
        if (cannon_arc.x == thid) {
            cannon_arcs[i] = cannon_arc.y;
            represent_count++;
        }
    }
    //Step 2: Scan and store in lookup table
    ex_scan<node_t>(smem,represent_count,N_f);
    
    if(thid < N_f){
    int arc_count = 0;
    for (size_t i = 0; i < FD.face_degrees[thid]; ++i){
        if (cannon_arcs[i] != UINT16_MAX) {
            triangle_numbers[thid*MaxDegree + i] = smem[thid] + arc_count;
            ++arc_count;
        }
    }}

    __syncthreads(); //TODO check if this is needed
    //Step 3: Store the arcs representing each triangle in an ordered list (representative_arc_list)
    edge_t* representative_arc_list = reinterpret_cast<edge_t*>(smem);
    if(thid < N_f)
    for (size_t i = 0; i < FD.face_degrees[thid]; i++){
        if(cannon_arcs[i] != UINT16_MAX){
            auto idx = triangle_numbers[thid*MaxDegree + i];
            representative_arc_list[idx] = {node_t(thid), cannon_arcs[i]}; 
        }
    }
        
    __syncthreads();
    //Step 4: Find neighbouring triangles through their canonical arcs and find their IDs by looking up in triangle_numbers
    for(auto tix = threadIdx.x; tix < N_t; tix += blockDim.x){

        auto [u, v] = representative_arc_list[tix];
        node_t w = FD.next(u,v);

        edge_t edge_b = FD.get_cannonical_triangle_arc(v, u); G_out[isomer_idx*N_t*3 + tix*3 + 0] = triangle_numbers[edge_b.x * MaxDegree + FD.dedge_ix(edge_b.x, edge_b.y)];
        edge_t edge_c = FD.get_cannonical_triangle_arc(w, v); G_out[isomer_idx*N_t*3 + tix*3 + 1] = triangle_numbers[edge_c.x * MaxDegree + FD.dedge_ix(edge_c.x, edge_c.y)];
        edge_t edge_d = FD.get_cannonical_triangle_arc(u, w); G_out[isomer_idx*N_t*3 + tix*3 + 2] = triangle_numbers[edge_d.x * MaxDegree + FD.dedge_ix(edge_d.x, edge_d.y)];
    }
    }
}

template <int MaxDegree>
void dualise_V0(const CuArray<node_t>& G_in, const CuArray<uint8_t>& deg,  CuArray<node_t>& G_out , const int N_f,  const int N_graphs, const LaunchCtx& ctx, const LaunchPolicy policy){
    cudaSetDevice(ctx.get_device_id());
    size_t smem = sizeof(node_t)*N_f*(MaxDegree*2 + 2);
    auto N = 2*(N_f - 2);
    static int n_blocks = 0; 
    static bool first = true;
    if (first){ 
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_blocks, (void*)dualise_V0_<MaxDegree>, N, smem); 
        first = false;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        n_blocks *= prop.multiProcessorCount;
    }

    void* kargs[]{(void*)&G_in.data, (void*)&deg.data, (void*)&G_out.data, (void*)&N_f, (void*)&N_graphs};
    if (policy == LaunchPolicy::SYNC) cudaStreamSynchronize(ctx.stream);
        cudaLaunchCooperativeKernel((void*)dualise_V0_<MaxDegree>, n_blocks, N, kargs, smem, ctx.stream);
    if (policy == LaunchPolicy::SYNC) cudaStreamSynchronize(ctx.stream);
}

template <int MaxDegree>
void dualise_V1(const CuArray<node_t>& G_in, const CuArray<uint8_t>& deg,  CuArray<node_t>& G_out , const int N_f,  const int N_graphs, const LaunchCtx& ctx, const LaunchPolicy policy){
    cudaSetDevice(ctx.get_device_id());
    size_t smem = sizeof(node_t)*N_f*(MaxDegree*2 + 2);
    auto N = 2*(N_f - 2);
    int lcm = (N_f + 31) & ~31; //Round up to nearest multiple of 32 [Warp Size] (Least Common Multiple)
    static int n_blocks = 0; 
    static bool first = true;
    if (first){ 
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_blocks, (void*)dualise_V1_<MaxDegree>, lcm, smem); 
        first = false;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        n_blocks *= prop.multiProcessorCount;
    }

    void* kargs[]{(void*)&G_in.data, (void*)&deg.data, (void*)&G_out.data, (void*)&N_f, (void*)&N_graphs};
    if (policy == LaunchPolicy::SYNC) cudaStreamSynchronize(ctx.stream);
        cudaLaunchCooperativeKernel((void*)dualise_V1_<MaxDegree>, n_blocks, lcm, kargs, smem, ctx.stream);
    if (policy == LaunchPolicy::SYNC) cudaStreamSynchronize(ctx.stream);
}


template <int MaxDegree>
void dualise_V0(const node_t* G_in, const uint8_t* deg,  node_t* G_out , const int N_f,  const int N_graphs){
    size_t smem = sizeof(node_t)*N_f*(MaxDegree*2 + 2);
    static int n_blocks = 0; 
    static bool first = true;
    if (first){ cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_blocks, (void*)dualise_V0_<MaxDegree>, N_f, smem); first = false;}

    
    void* kargs[]{(void*)&G_in, (void*)&G_out, (void*)&N_f, (void*)&N_graphs};
}

template <int MaxDegree>
void dualise_V1(const node_t* G_in, const uint8_t* deg,  node_t* G_out , const int N_f,  const int N_graphs){
    size_t smem = sizeof(node_t)*N_f*(MaxDegree*2 + 2);
    int lcm = (N_f + 31) & ~31; //Round up to nearest multiple of 32 [Warp Size] (Least Common Multiple)
    static int n_blocks = 0; 
    static bool first = true;
    if (first){ cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_blocks, (void*)dualise_V1_<MaxDegree>, lcm, smem); first = false;}

    
    void* kargs[]{(void*)&G_in, (void*)&G_out, (void*)&N_f, (void*)&N_graphs};
}


//Note: Force the compiler to generate all the required template instantiations
int declare_generics(){
    dualise_V0<4>(nullptr,nullptr,nullptr,0,0);
    dualise_V0<5>(nullptr,nullptr,nullptr,0,0);
    dualise_V0<6>(nullptr,nullptr,nullptr,0,0);
    dualise_V0<7>(nullptr,nullptr,nullptr,0,0);
    dualise_V0<8>(nullptr,nullptr,nullptr,0,0);
    dualise_V0<9>(nullptr,nullptr,nullptr,0,0);
    dualise_V0<10>(nullptr,nullptr,nullptr,0,0);
    
    dualise_V1<4>(nullptr,nullptr,nullptr,0,0);
    dualise_V1<5>(nullptr,nullptr,nullptr,0,0);
    dualise_V1<6>(nullptr,nullptr,nullptr,0,0);
    dualise_V1<7>(nullptr,nullptr,nullptr,0,0);
    dualise_V1<8>(nullptr,nullptr,nullptr,0,0);
    dualise_V1<9>(nullptr,nullptr,nullptr,0,0);
    dualise_V1<10>(nullptr,nullptr,nullptr,0,0);
    



    CuArray<node_t> G_in(1); CuArray<uint8_t> deg(1); CuArray<node_t> G_out(1);
    dualise_V0<4>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V0<5>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V0<6>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V0<7>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V0<8>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V0<9>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V0<10>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);

    dualise_V1<4>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V1<5>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V1<6>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V1<7>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V1<8>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V1<9>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    dualise_V1<10>(G_in, deg, G_out, 0, 0, LaunchCtx(), LaunchPolicy::ASYNC);
    return 0;
}