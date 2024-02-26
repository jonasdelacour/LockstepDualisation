#include "sycl_kernels.h"
#include "numeric"
#include <vector>
#include <tuple>
#include <iterator>
#include <type_traits>
#define SYCLBATCH
#include <isomer_batch.h>
#include "forcefield-includes.cc"
using namespace sycl;
//Template specialisation for dualise
#define GRID_STRIDED 0
#define mod(i,n) i + ((i < 0) - (i >= n)) * n
#define UINT_TYPE_MAX std::numeric_limits<UINT_TYPE>::max()
#define EMPTY_NODE()  std::numeric_limits<node_t>::max()
#define CPU_WGS_VC 212
#define CPU_WGS_VT 108

template <int MaxDegree, typename T, typename K> class Dualise1 {};
template <int MaxDegree, typename T, typename K> class Dualise2 {};
template <int MaxDegree, typename T, typename K> class Dualise3_1 {};
template <int MaxDegree, typename T, typename K> class Dualise3_2 {};
template <int MaxDegree, typename T, typename K> class Dualise4 {};
template <int MaxDegree, typename T, typename K> class Dualise5 {};


template<int MaxDegree, typename K>
struct DeviceDualGraph{
    //Check that K is integral
    INT_TYPEDEFS(K);

  
    const K* dual_neighbours;                          //(Nf x MaxDegree)
    const K* face_degrees;                            //(Nf x 1)
    
    DeviceDualGraph(const K* dual_neighbours, const K* face_degrees) : dual_neighbours(dual_neighbours), face_degrees(face_degrees) {}

    K dedge_ix(const K u, const K v) const{
        K result;
        #pragma unroll
        for (int j = 0; j < MaxDegree; j++){
            result = (dual_neighbours[u*MaxDegree + j] == v) ? j : result;
        }
	    return result;
    }

    K get_node(const K u, const int idx) const{
        __builtin_assume(face_degrees[u] <= MaxDegree);
        //return dual_neighbours[u*MaxDegree + mod(idx, face_degrees[u])];
        return dual_neighbours[u*MaxDegree + mod(idx, face_degrees[u])];
    }

    /**
     * @brief returns the next node in the clockwise order around u
     * @param v the current node around u
     * @param u the node around which the search is performed
     * @return the next node in the clockwise order around u
     */
    K next(const K u, const K v) const{
        K j = dedge_ix(u,v);
        return get_node(u,j+1);
    }
    
    /**
     * @brief returns the prev node in the clockwise order around u
     * @param v the current node around u
     * @param u the node around which the search is performed
     * @return the previous node in the clockwise order around u
     */
    K prev(const K u, const K v) const{
        K j = dedge_ix(u,v);
        return get_node(u, j-1);
    }

    /**
     * @brief Find the node that comes next on the face. given by the edge (u,v)
     * @param u Source of the edge.
     * @param v Destination node.
     * @return The node that comes next on the face.
     */
    K next_on_face(const K u, const K v) const{
        return prev(v,u);
    }

    /**
     * @brief Find the node that comes next on the face. given by the edge (u,v)
     * @param u Source of the edge.
     * @param v Destination node.
     * @return The node that comes next on the face.
     */
    K prev_on_face(const K u, const K v) const{
        return next(v,u);
    }

    node2 get_cannonical_triangle_arc_from_ix(const K u, const K v_idx) const{
        //In a triangle u, v, w there are only 3 possible representative arcs, the cannonical arc is chosen as the one with the smalles source node.
        K v = dual_neighbours[u*MaxDegree + v_idx];
        K w = get_node(u, v_idx+1);
        return (u < v) ? ((u < w) ? node2{u, v} : node2{w, u}) : ((v < w) ? node2{v, w} : node2{w, u});
    }

    node2 get_cannonical_triangle_arc(const K u, const K v) const{
        K w = next(u,v);
        return (u < v) ? ((u < w) ? node2{u, v} : node2{w, u}) : ((v < w) ? node2{v, w} : node2{w, u});
    }

    //Assumes that the triangle is directed u -> v -> w
    node2 get_cannonical_triangle_arc(const K u, const K v, const K w) const{
        return (u < v) ? ((u < w) ? node2{u, v} : node2{w, u}) : ((v < w) ? node2{v, w} : node2{w, u});
    }    
};

int roundUp(int numToRound, int multiple) 
{
    assert(multiple);
    return ((numToRound + multiple - 1) / multiple) * multiple;
}

int get_best_work_group_size(int numToRound, const std::string& kernel_name, const sycl::queue& Q){
    if(Q.get_device().is_gpu()) return roundUp(numToRound, Q.get_device().get_info<info::device::sub_group_sizes>()[0]);
    kernel_bundle kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Q.get_context());
    std::vector<kernel_id> k_ids = kb.get_kernel_ids();
    auto preferred_wg_size = 0; 
    for(auto k_id : k_ids){
        auto kernel = kb.get_kernel(k_id);
        if (kernel.get_info<info::kernel::function_name>().find(kernel_name) != std::string::npos){
            preferred_wg_size = kernel.get_info<info::kernel_device_specific::preferred_work_group_size_multiple>(Q.get_device());
        }
    }
    return roundUp(numToRound, preferred_wg_size);
}

template <int MaxDegree, typename T, typename K>
void dualise_sycl_v4(sycl::queue&Q, IsomerBatch<T,K>& batch, const LaunchPolicy policy){
    INT_TYPEDEFS(K);
    if(policy == LaunchPolicy::SYNC) Q.wait();
    Q.submit([&](handler &h) {
        auto N = batch.n_atoms;
        auto Nf = batch.n_faces;
        auto WGS = Q.get_device().is_cpu() ? CPU_WGS_VC : N;
        auto capacity = batch.m_capacity;
        
        //Create local accessors to shared memory
        local_accessor<node_t, 1>   triangle_numbers(Nf*MaxDegree, h);
        local_accessor<node_t, 1>   cached_neighbours(Nf*MaxDegree, h);
        local_accessor<node_t, 1>   cached_degrees(Nf, h);
        local_accessor<node2, 1>    arc_list(N, h);

        auto num_compute_units = Q.get_device().get_info<info::device::max_compute_units>();
        auto n_blocks_strided = num_compute_units;
        //Create device accessors
        accessor     cubic_neighbours_dev(batch.cubic_neighbours, h, write_only);
        accessor     face_degrees_dev(batch.face_degrees, h, read_only);
        accessor     dual_neighbours_dev(batch.dual_neighbours, h, read_only);
        /* 
        std::cout << N * capacity << std::endl; */
        #if GRID_STRIDED
        h.parallel_for<Dualise4<MaxDegree,T,K>>(sycl::nd_range(sycl::range{N*n_blocks_strided}, sycl::range{N}), [=](nd_item<1> nditem) {
        #else
        h.parallel_for<Dualise4<MaxDegree,T,K>>(sycl::nd_range(sycl::range{WGS*capacity}, sycl::range{WGS}), [=](nd_item<1> nditem) {
        #endif

            auto cta = nditem.get_group();
            auto thid = nditem.get_local_linear_id();
            auto bid = nditem.get_group_linear_id();
            #if GRID_STRIDED == 0
            auto isomer_idx = bid;
            #endif
            #if GRID_STRIDED
            for (size_t isomer_idx = bid; isomer_idx < capacity; isomer_idx += n_blocks_strided)
            {
            #endif
            
            cta.async_work_group_copy(cached_neighbours.get_pointer(), dual_neighbours_dev.get_pointer() + bid*Nf*MaxDegree, Nf*MaxDegree);
            cta.async_work_group_copy(cached_degrees.get_pointer(),    face_degrees_dev.get_pointer() + bid*Nf, Nf);
            DeviceDualGraph<MaxDegree, node_t> FD(cached_neighbours.get_pointer(), cached_degrees.get_pointer());
        
            node_t cannon_arcs[MaxDegree]; for(size_t i=0;i<MaxDegree;i++) cannon_arcs[i] = EMPTY_NODE(); // memset sets bytes, but node_t is multi-byte.
            node_t rep_count  = 0;
            sycl::group_barrier(cta);     

            if (thid < Nf){
                for (node_t i = 0; i < FD.face_degrees[thid]; i++){
                    node2 cannon_arc = FD.get_cannonical_triangle_arc(thid, FD.dual_neighbours[thid*MaxDegree + i], FD.get_node(thid, i+1));
                    if (cannon_arc[0] == thid){
                        cannon_arcs[i] = cannon_arc[1];
                        rep_count++;
                    }
                }
            }

            node_t scan_result = exclusive_scan_over_group(cta, rep_count, plus<node_t>{});

            if (thid < Nf){
                node_t arc_count = 0;
                for (node_t i = 0; i < FD.face_degrees[thid]; i++){
                    if(cannon_arcs[i] != EMPTY_NODE()){
                        auto idx = scan_result + arc_count;
                        triangle_numbers[thid*MaxDegree + i] = idx;
                        arc_list[idx] = {node_t(thid), i};
                        ++arc_count;
                    }    
                }
            }
            sycl::group_barrier(cta);
//          
            if(thid < N){
            auto [u, v_idx] = arc_list[thid];
            auto v =        FD.dual_neighbours[u*MaxDegree + v_idx];
            auto uv_prev =  FD.get_node(u, int(v_idx)-1);
            auto w =        FD.get_node(u, v_idx+1);
            auto uw_next =  FD.get_node(u, v_idx+2);

            auto edge_b = FD.get_cannonical_triangle_arc(v, u, uv_prev); cubic_neighbours_dev[isomer_idx*N*3 + thid*3 + 0] = triangle_numbers[edge_b[0]*MaxDegree + FD.dedge_ix(edge_b[0], edge_b[1])];
            auto edge_c = FD.get_cannonical_triangle_arc(w, v);          cubic_neighbours_dev[isomer_idx*N*3 + thid*3 + 1] = triangle_numbers[edge_c[0]*MaxDegree + FD.dedge_ix(edge_c[0], edge_c[1])];
            auto edge_d = FD.get_cannonical_triangle_arc(u, w, uw_next); cubic_neighbours_dev[isomer_idx*N*3 + thid*3 + 2] = triangle_numbers[edge_d[0]*MaxDegree + FD.dedge_ix(edge_d[0], edge_d[1])];
            }

            #if GRID_STRIDED
            }
            #endif
        });
    });
    if(policy == LaunchPolicy::SYNC) Q.wait();
}


template <int MaxDegree, typename T, typename K>
void dualise_sycl_v1(sycl::queue&Q, IsomerBatch<T,K>& batch, const LaunchPolicy policy){
    INT_TYPEDEFS(K);
    
    if(policy == LaunchPolicy::SYNC) Q.wait();
    auto subgroup_size = Q.get_device().get_info<info::device::sub_group_sizes>()[0];
    size_t lcm = roundUp(batch.n_faces, subgroup_size); 
    if (Q.get_device().is_cpu()) lcm = CPU_WGS_VT;
    Q.submit([&](handler &h) {
        auto N = batch.n_atoms;
        auto Nf = batch.n_faces;
        auto capacity = batch.m_capacity;
        
        //Create local accessors to shared memory
        local_accessor<node_t, 1>   triangle_numbers(Nf*MaxDegree, h);
        local_accessor<node_t, 1>   cached_neighbours(Nf*MaxDegree, h);
        local_accessor<node_t, 1>   cached_degrees(Nf, h);
        local_accessor<node2, 1>    arc_list(N, h);
        auto num_compute_units = Q.get_device().get_info<info::device::max_compute_units>();
        auto n_blocks_strided = num_compute_units;
        //Create device accessors
        accessor     cubic_neighbours_dev(batch.cubic_neighbours, h, write_only);
        accessor     face_degrees_dev(batch.face_degrees, h, read_only);
        accessor     dual_neighbours_dev(batch.dual_neighbours, h, read_only);
        /* 
        std::cout << N * capacity << std::endl; */
        #if GRID_STRIDED
        h.parallel_for<Dualise1<MaxDegree,T,K>>(sycl::nd_range(sycl::range{lcm*n_blocks_strided}, sycl::range{lcm}), [=](nd_item<1> nditem) {
        #else
        h.parallel_for<Dualise1<MaxDegree,T,K>>(sycl::nd_range(sycl::range{lcm*capacity}, sycl::range{lcm}), [=](nd_item<1> nditem) {
        #endif
            auto cta = nditem.get_group();
            auto thid = nditem.get_local_linear_id();
            auto bid = nditem.get_group_linear_id();
            #if GRID_STRIDED == 0
            auto isomer_idx = bid;
            #endif
            #if GRID_STRIDED
            for (size_t isomer_idx = bid; isomer_idx < capacity; isomer_idx += n_blocks_strided)
            {
            #endif
            cta.async_work_group_copy(cached_neighbours.get_pointer(), dual_neighbours_dev.get_pointer() + bid*Nf*MaxDegree, Nf*MaxDegree);
            cta.async_work_group_copy(cached_degrees.get_pointer(), face_degrees_dev.get_pointer() + bid*Nf, Nf);
            
            DeviceDualGraph<MaxDegree, node_t> FD(cached_neighbours.get_pointer(), cached_degrees.get_pointer());
            node_t cannon_ixs[MaxDegree];
            node_t rep_count  = 0;
            sycl::group_barrier(cta);     
            if (thid < Nf){
                for (int i = 0; i < FD.face_degrees[thid]; i++){
                    if(thid < FD.dual_neighbours[thid*MaxDegree + i] && thid < FD.get_node(thid, i+1)){
                        cannon_ixs[rep_count] = i;
                        rep_count++;
                    }
                }
            }

            node_t scan_result = exclusive_scan_over_group(cta, rep_count, plus<node_t>{});

            if (thid < Nf){
                for (int ii = 0; ii < rep_count; ii++){
                    auto idx = scan_result + ii;
                    triangle_numbers[thid*MaxDegree + cannon_ixs[ii]] = idx;
                    arc_list[idx] = {node_t(thid), cannon_ixs[ii]};
                }
            }
            sycl::group_barrier(cta);

            for(int tix = thid; tix < N; tix += lcm){
                auto [u, v_idx] = arc_list[tix];
                auto v =        FD.dual_neighbours[u*MaxDegree + v_idx];
                auto uv_prev =  FD.get_node(u, int(v_idx)-1);
                auto w =        FD.get_node(u, v_idx+1);
                auto uw_next =  FD.get_node(u, v_idx+2);

                auto edge_b = FD.get_cannonical_triangle_arc(v, u, uv_prev);    cubic_neighbours_dev[isomer_idx*N*3 + tix*3 + 0] = triangle_numbers[edge_b[0]*MaxDegree + FD.dedge_ix(edge_b[0], edge_b[1])];
                auto edge_c = FD.get_cannonical_triangle_arc(w, v);             cubic_neighbours_dev[isomer_idx*N*3 + tix*3 + 1] = triangle_numbers[edge_c[0]*MaxDegree + FD.dedge_ix(edge_c[0], edge_c[1])];
                auto edge_d = FD.get_cannonical_triangle_arc(u, w, uw_next);    cubic_neighbours_dev[isomer_idx*N*3 + tix*3 + 2] = triangle_numbers[edge_d[0]*MaxDegree + FD.dedge_ix(edge_d[0], edge_d[1])];
            }

            #if GRID_STRIDED
            }
            #endif
        });
    });
    if(policy == LaunchPolicy::SYNC) Q.wait();
}

template <int MaxDegree, typename T, typename K>
void dualise_sycl_v2(sycl::queue&Q, IsomerBatch<T,K>& batch, const LaunchPolicy policy){
    INT_TYPEDEFS(K);
    
    auto WGS = Q.get_device().is_cpu() ? CPU_WGS_VT: batch.n_faces;
    if(policy == LaunchPolicy::SYNC) Q.wait();
    Q.submit([&](handler &h) {
        auto N = batch.n_atoms;
        auto Nf = batch.n_faces;
        auto capacity = batch.m_capacity;
        
        //Create local accessors to shared memory
        local_accessor<node_t, 1>   triangle_numbers(Nf*MaxDegree, h);
        local_accessor<node_t, 1>   cached_neighbours(Nf*MaxDegree, h);
        local_accessor<node_t, 1>   cached_degrees(Nf, h);
        local_accessor<node2, 1>    arc_list(N, h);

        auto num_compute_units = Q.get_device().get_info<info::device::max_compute_units>();
        auto n_blocks_strided = num_compute_units;
        //Create device accessors
        accessor     cubic_neighbours_dev(batch.cubic_neighbours, h, write_only);
        accessor     face_degrees_dev(batch.face_degrees, h, read_only);
        accessor     dual_neighbours_dev(batch.dual_neighbours, h, read_only);
        /* 
        std::cout << N * capacity << std::endl; */
        #if GRID_STRIDED
        h.parallel_for<Dualise2<MaxDegree,T,K>>(sycl::nd_range(sycl::range{Nf*n_blocks_strided}, sycl::range{N}), [=](nd_item<1> nditem) {
        #else
        h.parallel_for<Dualise2<MaxDegree,T,K>>(sycl::nd_range(sycl::range{WGS*capacity}, sycl::range{WGS}), [=](nd_item<1> nditem) {
        #endif
            auto cta = nditem.get_group();
            auto thid = nditem.get_local_linear_id();
            auto bid = nditem.get_group_linear_id();
            #if GRID_STRIDED == 0
            auto isomer_idx = bid;
            #endif
            #if GRID_STRIDED
            for (size_t isomer_idx = bid; isomer_idx < capacity; isomer_idx += n_blocks_strided)
            {
            #endif
            
            cta.async_work_group_copy(cached_neighbours.get_pointer(), dual_neighbours_dev.get_pointer() + bid*Nf*MaxDegree, Nf*MaxDegree);
            cta.async_work_group_copy(cached_degrees.get_pointer(),    face_degrees_dev.get_pointer() + bid*Nf, Nf);
            
            DeviceDualGraph<MaxDegree, node_t> FD(cached_neighbours.get_pointer(), cached_degrees.get_pointer());

            node_t cannon_arcs[MaxDegree]; for(size_t i=0;i<MaxDegree;i++) cannon_arcs[i] = EMPTY_NODE(); // memset sets bytes, but node_t is multi-byte.
            node_t rep_count  = 0;
            sycl::group_barrier(cta);     
            if (thid < Nf)
            for (node_t i = 0; i < FD.face_degrees[thid]; i++){
                node2 cannon_arc = FD.get_cannonical_triangle_arc(thid, FD.dual_neighbours[thid*MaxDegree + i], FD.get_node(thid, i+1));
                if (cannon_arc[0] == thid){
                    cannon_arcs[i] = cannon_arc[1];
                    rep_count++;
                }
            }

            node_t scan_result = exclusive_scan_over_group(cta, rep_count, plus<node_t>{});

            if (thid < Nf){
                node_t arc_count = 0;
                for (node_t i = 0; i < FD.face_degrees[thid]; i++){
                    if(cannon_arcs[i] != EMPTY_NODE()){
                        auto idx = scan_result + arc_count;
                        triangle_numbers[thid*MaxDegree + i] = idx;
                        //arc_list[idx] = {node_t(thid), cannon_arcs[i]};
                        ++arc_count;
                    }    
                }
            }

            sycl::group_barrier(cta);
            if(thid < Nf)
            for(node_t i = 0; i < FD.face_degrees[thid]; i++){
                if(cannon_arcs[i] != EMPTY_NODE()){
                    auto idx = triangle_numbers[thid*MaxDegree + i];
                    auto [u, v] = node2{node_t(thid), cannon_arcs[i]};
                    auto uv_prev =  FD.get_node(u, int(i)-1);
                    auto w =        FD.get_node(u, i+1);
                    auto uw_next =  FD.get_node(u, i+2);

                    auto edge_b = FD.get_cannonical_triangle_arc(v, u, uv_prev); cubic_neighbours_dev[isomer_idx*N*3 + idx*3 + 0] = triangle_numbers[edge_b[0]*MaxDegree + FD.dedge_ix(edge_b[0], edge_b[1])];
                    auto edge_c = FD.get_cannonical_triangle_arc(w, v); cubic_neighbours_dev[isomer_idx*N*3 + idx*3 + 1] = triangle_numbers[edge_c[0]*MaxDegree + FD.dedge_ix(edge_c[0], edge_c[1])];
                    auto edge_d = FD.get_cannonical_triangle_arc(u, w, uw_next); cubic_neighbours_dev[isomer_idx*N*3 + idx*3 + 2] = triangle_numbers[edge_d[0]*MaxDegree + FD.dedge_ix(edge_d[0], edge_d[1])];
                }
            }

            #if GRID_STRIDED
            }
            #endif
        });
    });
    if(policy == LaunchPolicy::SYNC) Q.wait();
}

template <int MaxDegree, typename T>
struct DualBuffers{
    std::vector<sycl::device> devices;

    std::vector<sycl::buffer<T,1>> triangle_numbers;
    std::vector<sycl::buffer<T,1>> arc_list;
    std::vector<std::array<size_t, 2>> dims; //natoms, capacity

    //SYCL lacks a find for retrieving a unique identifier for a given queue, platform, or device so we have to do this manually
    int get_device_index(const sycl::device& device){
        for(int i = 0; i < devices.size(); i++){
            if(devices[i] == device){
                return i;
            }
        }
        return -1;
    }

    DualBuffers(size_t N, size_t capacity, const sycl::device& device){
        auto Nf = N / 2 + 2;
        devices = {device};
        triangle_numbers = std::vector<sycl::buffer<T,1>>(1, sycl::buffer<T,1>(Nf*MaxDegree*capacity));
        arc_list = std::vector<sycl::buffer<T,1>>(1, sycl::buffer<T,1>(N*capacity*2));
        dims = std::vector<std::array<size_t, 2>>(1, {N, capacity});
    }

    void resize(size_t N, size_t Nf, size_t capacity, size_t idx){
        triangle_numbers[idx]   = sycl::buffer<T,1>(Nf*MaxDegree*capacity);
        arc_list[idx]      = sycl::buffer<T,1>(N*capacity*2);
        dims[idx]             = {N, capacity};
    }
    
    void append_buffers(size_t N,size_t Nf, size_t capacity, const sycl::device& device){
        triangle_numbers.push_back(sycl::buffer<T,1>(Nf*MaxDegree*capacity));
        arc_list.push_back(sycl::buffer<T,1>(N*capacity*2));
        dims.push_back({N, capacity});
        devices.push_back(device);
    }

    void update(size_t N, size_t capacity, const sycl::device& device){
        auto idx = get_device_index(device);
        auto Nf = N / 2 + 2;
        if(idx == -1){
            append_buffers(N, Nf, capacity,device);
            std::cout << "Appended buffers for device " << device.get_info<sycl::info::device::name>() << " ID: " << devices.size() - 1 << "\n";
        }
        else if (N != dims[idx][0] || capacity != dims[idx][1]){
            resize(N, Nf, capacity, idx);
            std::cout << "Resized buffers for device " << device.get_info<sycl::info::device::name>() << " ID: " << idx << "\n";
        }
    }
};

template <int MaxDegree, typename T, typename K>
void dualise_sycl_v3(sycl::queue&Q, IsomerBatch<T,K>& batch, const LaunchPolicy policy){
    INT_TYPEDEFS(K);
    static DualBuffers<MaxDegree, K> buffers(batch.n_atoms, batch.m_capacity, Q.get_device());
    buffers.update(batch.n_atoms, batch.m_capacity, Q.get_device());
    auto d_idx = buffers.get_device_index(Q.get_device());
    auto WGS1 = Q.get_device().is_cpu() ? CPU_WGS_VT: batch.n_faces;
    auto WGS2 = Q.get_device().is_cpu() ? CPU_WGS_VC: batch.n_atoms;
    if(policy == LaunchPolicy::SYNC) Q.wait();
    Q.submit([&](handler &h) {
        auto N = batch.n_atoms;
        auto Nf = batch.n_faces;
        auto capacity = batch.m_capacity;
        
        //Create local accessors to shared memory
        local_accessor<node_t, 1>   cached_neighbours(Nf*MaxDegree, h);
        local_accessor<node_t, 1>   cached_degrees(Nf, h);

        auto num_compute_units = Q.get_device().get_info<info::device::max_compute_units>();
        auto n_blocks_strided = num_compute_units;
        //Create device accessors
        accessor     face_degrees_dev       (batch.face_degrees, h, read_only);
        accessor     dual_neighbours_dev    (batch.dual_neighbours, h, read_only);
        accessor     triangle_numbers_dev   (buffers.triangle_numbers[d_idx], h, write_only);
        accessor     arc_list_dev           (buffers.arc_list[d_idx], h, write_only);
        /* 
        std::cout << N * capacity << std::endl; */
        #if GRID_STRIDED
        h.parallel_for<Dualise3_1<MaxDegree,T,K>>(sycl::nd_range(sycl::range{Nf*n_blocks_strided}, sycl::range{N}), [=](nd_item<1> nditem) {
        #else
        h.parallel_for<Dualise3_1<MaxDegree,T,K>>(sycl::nd_range(sycl::range{WGS1*capacity}, sycl::range{WGS1}), [=](nd_item<1> nditem) {
        #endif
            auto cta = nditem.get_group();
            auto thid = nditem.get_local_linear_id();
            auto bid = nditem.get_group_linear_id();
            #if GRID_STRIDED == 0
            auto isomer_idx = bid;
            #endif
            #if GRID_STRIDED
            for (size_t isomer_idx = bid; isomer_idx < capacity; isomer_idx += n_blocks_strided)
            {
            #endif
            
            auto event1 = cta.async_work_group_copy(cached_neighbours.get_pointer(), dual_neighbours_dev.get_pointer() + bid*Nf*MaxDegree, Nf*MaxDegree);
            auto event2 = cta.async_work_group_copy(cached_degrees.get_pointer(),    face_degrees_dev.get_pointer() + bid*Nf, Nf);
            
            cta.wait_for(event1, event2);

            DeviceDualGraph<MaxDegree, node_t> FD(cached_neighbours.get_pointer(), cached_degrees.get_pointer());

            node_t cannon_arcs[MaxDegree]; for(size_t i=0;i<MaxDegree;i++) cannon_arcs[i] = EMPTY_NODE(); // memset sets bytes, but node_t is multi-byte.
            node_t rep_count  = 0;
            sycl::group_barrier(cta);     

            if (thid < Nf)
            for (node_t i = 0; i < FD.face_degrees[thid]; i++){
                node2 cannon_arc = FD.get_cannonical_triangle_arc(thid, FD.dual_neighbours[thid*MaxDegree + i], FD.get_node(thid, i+1));
                if (cannon_arc[0] == thid){
                    cannon_arcs[i] = cannon_arc[1];
                    rep_count++;
                }
            }

            node_t scan_result = exclusive_scan_over_group(cta, rep_count, plus<node_t>{});

            node_t arc_count = 0;
            if (thid < Nf)
            for (node_t i = 0; i < FD.face_degrees[thid]; i++){
                if(cannon_arcs[i] != EMPTY_NODE()){
                    auto idx = scan_result + arc_count;
                    triangle_numbers_dev[thid*MaxDegree + i + isomer_idx* Nf*MaxDegree] = idx;
                    arc_list_dev[2*(idx + isomer_idx*N)] = node_t(thid);
                    arc_list_dev[2*(idx + isomer_idx*N) + 1] = i;
                    ++arc_count;
                }    
            }
            #if GRID_STRIDED
            }
            #endif
        });
    });

    Q.submit([&](handler &h) {
        auto N = batch.n_atoms;
        auto Nf = batch.n_faces;
        auto capacity = batch.m_capacity;
        
        //Create local accessors to shared memory
        local_accessor<node_t, 1>   triangle_numbers_local(Nf*MaxDegree, h);
        local_accessor<node_t, 1>   cached_neighbours(Nf*MaxDegree, h);
        local_accessor<node_t, 1>   cached_degrees(Nf, h);
        local_accessor<node_t, 1>   arc_list_local(2*N, h);

        auto num_compute_units = Q.get_device().get_info<info::device::max_compute_units>();
        auto n_blocks_strided = num_compute_units;
        //Create device accessors
        accessor     cubic_neighbours_dev   (batch.cubic_neighbours, h, write_only);
        accessor     face_degrees_dev       (batch.face_degrees, h, read_only);
        accessor     dual_neighbours_dev    (batch.dual_neighbours, h, read_only);
        accessor     triangle_numbers_dev   (buffers.triangle_numbers[d_idx], h, read_only);
        accessor     arc_list_dev           (buffers.arc_list[d_idx], h, read_only);
        /* 
        std::cout << N * capacity << std::endl; */
        #if GRID_STRIDED
        h.parallel_for<Dualise3_2<MaxDegree,T,K>>(sycl::nd_range(sycl::range{N*n_blocks_strided}, sycl::range{N}), [=](nd_item<1> nditem) {
        #else
        h.parallel_for<Dualise3_2<MaxDegree,T,K>>(sycl::nd_range(sycl::range{WGS2*capacity}, sycl::range{WGS2}), [=](nd_item<1> nditem) {
        #endif
            auto cta = nditem.get_group();
            auto thid = nditem.get_local_linear_id();
            auto bid = nditem.get_group_linear_id();
            #if GRID_STRIDED == 0
            auto isomer_idx = bid;
            #endif
            #if GRID_STRIDED
            for (size_t isomer_idx = bid; isomer_idx < capacity; isomer_idx += n_blocks_strided)
            {
            #endif
            
            cta.async_work_group_copy(cached_neighbours.get_pointer(),          dual_neighbours_dev.get_pointer() + isomer_idx*Nf*MaxDegree, Nf*MaxDegree);
            cta.async_work_group_copy(cached_degrees.get_pointer(),             face_degrees_dev.get_pointer() + isomer_idx*Nf, Nf);
            cta.async_work_group_copy(triangle_numbers_local.get_pointer(),     triangle_numbers_dev.get_pointer() + isomer_idx*Nf * MaxDegree, Nf*MaxDegree);
            cta.async_work_group_copy(arc_list_local.get_pointer(),             arc_list_dev.get_pointer() + 2*isomer_idx*N, 2*N);
            
            DeviceDualGraph<MaxDegree, node_t> FD(cached_neighbours.get_pointer(), cached_degrees.get_pointer());

            sycl::group_barrier(cta);
            if (thid < N){
            auto u = arc_list_local[2*thid];
            auto v_idx = arc_list_local[2*thid + 1];
            auto v =        FD.dual_neighbours[u*MaxDegree + v_idx];
            auto w =       FD.get_node(u, v_idx+1);
            auto uv_prev =  FD.get_node(u, int(v_idx)-1);
            auto uw_next =  FD.get_node(u, v_idx+2);

            auto edge_b = FD.get_cannonical_triangle_arc(v, u, uv_prev); cubic_neighbours_dev[isomer_idx*N*3 + thid*3 + 0] = triangle_numbers_local[edge_b[0]*MaxDegree + FD.dedge_ix(edge_b[0], edge_b[1])];
            auto edge_c = FD.get_cannonical_triangle_arc(w, v); cubic_neighbours_dev[isomer_idx*N*3 + thid*3 + 1] = triangle_numbers_local[edge_c[0]*MaxDegree + FD.dedge_ix(edge_c[0], edge_c[1])];
            auto edge_d = FD.get_cannonical_triangle_arc(u, w, uw_next); cubic_neighbours_dev[isomer_idx*N*3 + thid*3 + 2] = triangle_numbers_local[edge_d[0]*MaxDegree + FD.dedge_ix(edge_d[0], edge_d[1])];
            }

            #if GRID_STRIDED
            }
            #endif
        });
    });
    if(policy == LaunchPolicy::SYNC) Q.wait();
}
        

template void dualise_sycl_v1<6, float,uint8_t>(sycl::queue&Q, IsomerBatch<float,uint8_t>& batch, const LaunchPolicy policy);
template void dualise_sycl_v2<6, float,uint8_t>(sycl::queue&Q, IsomerBatch<float,uint8_t>& batch, const LaunchPolicy policy);
template void dualise_sycl_v3<6, float,uint8_t>(sycl::queue&Q, IsomerBatch<float,uint8_t>& batch, const LaunchPolicy policy);
template void dualise_sycl_v4<6, float,uint8_t>(sycl::queue&Q, IsomerBatch<float,uint8_t>& batch, const LaunchPolicy policy);

template void dualise_sycl_v1<6, float,uint16_t>(sycl::queue&Q, IsomerBatch<float,uint16_t>& batch, const LaunchPolicy policy);
template void dualise_sycl_v2<6, float,uint16_t>(sycl::queue&Q, IsomerBatch<float,uint16_t>& batch, const LaunchPolicy policy);
template void dualise_sycl_v3<6, float,uint16_t>(sycl::queue&Q, IsomerBatch<float,uint16_t>& batch, const LaunchPolicy policy);
template void dualise_sycl_v4<6, float,uint16_t>(sycl::queue&Q, IsomerBatch<float,uint16_t>& batch, const LaunchPolicy policy);

template void dualise_sycl_v1<6, float,uint32_t>(sycl::queue&Q, IsomerBatch<float,uint32_t>& batch, const LaunchPolicy policy);
template void dualise_sycl_v2<6, float,uint32_t>(sycl::queue&Q, IsomerBatch<float,uint32_t>& batch, const LaunchPolicy policy);
template void dualise_sycl_v3<6, float,uint32_t>(sycl::queue&Q, IsomerBatch<float,uint32_t>& batch, const LaunchPolicy policy);
template void dualise_sycl_v4<6, float,uint32_t>(sycl::queue&Q, IsomerBatch<float,uint32_t>& batch, const LaunchPolicy policy);
