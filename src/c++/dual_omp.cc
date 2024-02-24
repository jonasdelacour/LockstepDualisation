#include "cpp_kernels.h"
#include <vector>
#include <iostream>
#include <limits>

template void dualise_omp_shared<6> (IsomerBatch<float,uint16_t>& B);
template void dualise_omp_shared<6> (IsomerBatch<float,uint8_t>& B);
template void dualise_omp_task<6>   (IsomerBatch<float,uint16_t>& B);
template void dualise_omp_task<6>   (IsomerBatch<float,uint8_t>& B);

template <int MaxDegree, typename K>
struct GraphWrapper{
    typedef std::pair<K,K> arc_t;   
    const K* neighbours;
    const K* degrees;

    GraphWrapper(const K* neighbours, const K* degrees) : neighbours(neighbours), degrees(degrees) {}

    K dedge_ix(const K u, const K v) const{
        for (K j = 0; j < degrees[u]; j++){
            if (neighbours[u*MaxDegree + j] == v) return j;
            if(j == degrees[u] - 1) abort();
        }
        return -1; //Should never get here.
    }

    K next(const K u, const K v) const{
        K j = dedge_ix(u,v);
        return neighbours[u*MaxDegree + ((j+1)%degrees[u])];
    }

    K prev(const K u, const K v) const{
        K j = dedge_ix(u,v);
        return neighbours[u*MaxDegree + ((j-1+degrees[u])%degrees[u])];
    }

    arc_t canon_arc(const K u, const K v) const{
        arc_t edge = {u,v};
        K w = next(u,v);
        if (v < u && v < w) return {v,w};
        if (w < u && w < v) return {w,u};
        return edge;
    }

};
template<int MaxDegree, typename T, typename K>
void dualise_omp_shared(IsomerBatch<T,K>& B){
    typedef std::pair<K,K> arc_t;
    auto N = B.n_atoms;
    auto Nf = B.n_faces;
    auto N_graphs = B.m_capacity;
    std::vector<K> triangle_numbers(MaxDegree*Nf, std::numeric_limits<K>::max());
    std::vector<K> canon_arcs(MaxDegree*Nf, std::numeric_limits<K>::max());
    std::vector<K> n_triangles(Nf, 0); //Number of triangles that each face owns.
    std::vector<K> scan_array(Nf, 0); //Scan array for prefix sum.
    std::vector<arc_t> triangle_arcs(N);
    #pragma omp parallel
    {
        for (size_t i = 0; i < N_graphs; ++i){
            GraphWrapper<MaxDegree, K> G(B.dual_neighbours.data() + i*Nf*MaxDegree, B.face_degrees.data() + i*Nf);
            #pragma omp for schedule(auto)
            for (size_t j = 0; j < Nf; ++j){
                n_triangles[j] = 0;
                for (size_t k = 0; k < G.degrees[j]; ++k){
                    arc_t carc = G.canon_arc(j, G.neighbours[j*MaxDegree + k]);
                    if(carc.first == j){
                        canon_arcs[j*MaxDegree + k] = carc.second;
                        n_triangles[j]++;
                    } else {
                        canon_arcs[j*MaxDegree + k] = std::numeric_limits<K>::max();
                    }
                }
            }
            #pragma omp barrier

            K accumulator = 0;
	    //            #pragma omp simd reduction(inscan,+:accumulator)
            for (size_t j = 0; j < Nf; ++j){
                scan_array[j] = accumulator;
		//  #pragma omp scan exclusive(accumulator)
                accumulator += n_triangles[j];
            }
            #pragma omp barrier
            #pragma omp for schedule(auto)
            for (size_t j = 0; j < Nf; ++j){
                uint8_t n = 0;
                //#pragma omp simd
                for (size_t k = 0; k < G.degrees[j]; ++k){
                    if (canon_arcs[j*MaxDegree + k] != std::numeric_limits<K>::max()){
                        triangle_numbers[j*MaxDegree + k] = scan_array[j] + n;
                        n++;
                        triangle_arcs[triangle_numbers[j*MaxDegree + k]] = {j,canon_arcs[j*MaxDegree + k]};
                    }
                }
            }
            #pragma omp barrier
            #pragma omp for schedule(auto)
            for (size_t j = 0; j < N; j++){
                K u = triangle_arcs[j].first;
                K v = triangle_arcs[j].second;
                K w = G.next(u,v);
                arc_t arc_a = G.canon_arc(v,u);
                arc_t arc_b = G.canon_arc(w,v);
                arc_t arc_c = G.canon_arc(u,w);
                B.cubic_neighbours[i*N*3 + j*3 + 0] = triangle_numbers[arc_a.first*MaxDegree + G.dedge_ix(arc_a.first, arc_a.second)];
                B.cubic_neighbours[i*N*3 + j*3 + 1] = triangle_numbers[arc_b.first*MaxDegree + G.dedge_ix(arc_b.first, arc_b.second)];
                B.cubic_neighbours[i*N*3 + j*3 + 2] = triangle_numbers[arc_c.first*MaxDegree + G.dedge_ix(arc_c.first, arc_c.second)];
            }
        }
    }
}

template<int MaxDegree, typename T, typename K>
void dualise_omp_task(IsomerBatch<T,K>& B){
    typedef std::pair<K,K> arc_t;
    #pragma omp parallel 
    {
    auto N = B.n_atoms;
    auto Nf = B.n_faces;
    auto N_graphs = B.m_capacity;
    std::vector<K> triangle_numbers(MaxDegree*Nf, std::numeric_limits<K>::max());
    std::vector<arc_t> triangle_arcs(N);
    #pragma omp for schedule(auto)
    for (size_t i = 0; i < N_graphs; ++i){
        GraphWrapper<6, K> G(B.dual_neighbours.data() + i*Nf*MaxDegree, B.face_degrees.data() + i*Nf);
        K accumulator = 0;
        for (int j = 0; j < Nf; ++j){
            for (int k = 0; k < G.degrees[j]; ++k){
                arc_t carc = G.canon_arc(j, G.neighbours[j*MaxDegree + k]);
                if(carc.first == j){
                    triangle_numbers[j*MaxDegree + k] = accumulator;
                    triangle_arcs[accumulator] = {j,carc.second};
                    accumulator++;
                }
            }
        }
        for (int j = 0; j < N; j++){
            K u = triangle_arcs[j].first;
            K v = triangle_arcs[j].second;
            K w = G.next(u,v);
            arc_t arc_a = G.canon_arc(v,u);
            arc_t arc_b = G.canon_arc(w,v);
            arc_t arc_c = G.canon_arc(u,w);
            B.cubic_neighbours[i*N*3 + j*3 + 0] = triangle_numbers[arc_a.first*MaxDegree + G.dedge_ix(arc_a.first, arc_a.second)];
            B.cubic_neighbours[i*N*3 + j*3 + 1] = triangle_numbers[arc_b.first*MaxDegree + G.dedge_ix(arc_b.first, arc_b.second)];
            B.cubic_neighbours[i*N*3 + j*3 + 2] = triangle_numbers[arc_c.first*MaxDegree + G.dedge_ix(arc_c.first, arc_c.second)];
        }
    }   
    }
         
} 
