#include "dual.h"
#include <vector>
#include <iostream>
typedef std::pair<uint16_t,uint16_t> arc_t;

template void dualise_V2<6>(const std::vector<node_t>& G_in, const std::vector<uint8_t>& degrees, std::vector<node_t>& G_out , const int Nf,  const int N_graphs);
template void dualise_V3<6>(const std::vector<node_t>& G_in, const std::vector<uint8_t>& degrees, std::vector<node_t>& G_out , const int Nf,  const int N_graphs);

template <int MaxDegree>
struct GraphWrapper{
    const uint16_t* neighbours;
    const uint8_t* degrees;

    GraphWrapper(const uint16_t* neighbours, const uint8_t* degrees) : neighbours(neighbours), degrees(degrees) {}

    uint16_t dedge_ix(const uint16_t u, const uint16_t v) const{
        for (uint8_t j = 0; j < degrees[u]; j++){
            if (neighbours[u*MaxDegree + j] == v) return j;
            if(j == degrees[u] - 1) exit(1);
        }
    }

    uint16_t next(const uint16_t u, const uint16_t v) const{
        uint16_t j = dedge_ix(u,v);
        return neighbours[u*MaxDegree + ((j+1)%degrees[u])];
    }

    uint16_t prev(const uint16_t u, const uint16_t v) const{
        uint16_t j = dedge_ix(u,v);
        return neighbours[u*MaxDegree + ((j-1+degrees[u])%degrees[u])];
    }

    arc_t canon_arc(const uint16_t u, const uint16_t v) const{
        arc_t edge = {u,v};
        uint16_t w = next(u,v);
        if (v < u && v < w) return {v,w};
        if (w < u && w < v) return {w,u};
        return edge;
    }

};
template<int MaxDegree>
void dualise_V2(const std::vector<node_t>& G_in, const std::vector<uint8_t>& degrees, std::vector<node_t>& G_out , const int Nf,  const int N_graphs){
    int N = (Nf - 2)*2;
    std::vector<uint16_t> triangle_numbers(MaxDegree*Nf, UINT16_MAX);
    std::vector<uint16_t> canon_arcs(MaxDegree*Nf, UINT16_MAX);
    std::vector<uint16_t> n_triangles(Nf, 0); //Number of triangles that each face owns.
    std::vector<uint16_t> scan_array(Nf, 0); //Scan array for prefix sum.
    std::vector<arc_t> triangle_arcs(N);
    #pragma omp parallel
    {
        for (size_t i = 0; i < N_graphs; ++i){
            GraphWrapper<MaxDegree> G(G_in.data() + i*Nf*MaxDegree, degrees.data() + i*Nf);
            #pragma omp for schedule(auto)
            for (size_t j = 0; j < Nf; ++j){
                n_triangles[j] = 0;
                for (size_t k = 0; k < G.degrees[j]; ++k){
                    arc_t carc = G.canon_arc(j, G.neighbours[j*MaxDegree + k]);
                    if(carc.first == j){
                        canon_arcs[j*MaxDegree + k] = carc.second;
                        n_triangles[j]++;
                    } else {
                        canon_arcs[j*MaxDegree + k] = UINT16_MAX;
                    }
                }
            }
            #pragma omp barrier

            uint16_t accumulator = 0;
            #pragma omp simd reduction(inscan,+:accumulator)
            for (size_t j = 0; j < Nf; ++j){
                scan_array[j] = accumulator;
                #pragma omp scan exclusive(accumulator)
                accumulator += n_triangles[j];
            }
            #pragma omp barrier
            #pragma omp for schedule(auto)
            for (size_t j = 0; j < Nf; ++j){
                uint8_t n = 0;
                //#pragma omp simd
                for (size_t k = 0; k < G.degrees[j]; ++k){
                    if (canon_arcs[j*MaxDegree + k] != UINT16_MAX){
                        triangle_numbers[j*MaxDegree + k] = scan_array[j] + n;
                        n++;
                        triangle_arcs[triangle_numbers[j*MaxDegree + k]] = {j,canon_arcs[j*MaxDegree + k]};
                    }
                }
            }
            #pragma omp barrier
            #pragma omp for schedule(auto)
            for (size_t j = 0; j < N; j++){
                uint16_t u = triangle_arcs[j].first;
                uint16_t v = triangle_arcs[j].second;
                uint16_t w = G.next(u,v);
                arc_t arc_a = G.canon_arc(v,u);
                arc_t arc_b = G.canon_arc(w,v);
                arc_t arc_c = G.canon_arc(u,w);
                G_out[i*N*3 + j*3 + 0] = triangle_numbers[arc_a.first*MaxDegree + G.dedge_ix(arc_a.first, arc_a.second)];
                G_out[i*N*3 + j*3 + 1] = triangle_numbers[arc_b.first*MaxDegree + G.dedge_ix(arc_b.first, arc_b.second)];
                G_out[i*N*3 + j*3 + 2] = triangle_numbers[arc_c.first*MaxDegree + G.dedge_ix(arc_c.first, arc_c.second)];
            }
        }
    }
}

template<int MaxDegree>
void dualise_V3(const std::vector<node_t>& G_in, const std::vector<uint8_t>& degrees, std::vector<node_t>& G_out , const int Nf,  const int N_graphs){
    #pragma omp parallel 
    {
    int N = (Nf - 2)*2;
    std::vector<uint16_t> triangle_numbers(MaxDegree*Nf, UINT16_MAX);
    std::vector<arc_t> triangle_arcs(N);
    #pragma omp for schedule(auto)
    for (size_t i = 0; i < N_graphs; ++i){
        GraphWrapper<6> G(G_in.data() + i*Nf*MaxDegree, degrees.data() + i*Nf);
        uint16_t accumulator = 0;
        for (size_t j = 0; j < Nf; ++j){
            for (size_t k = 0; k < G.degrees[j]; ++k){
                arc_t carc = G.canon_arc(j, G.neighbours[j*MaxDegree + k]);
                if(carc.first == j){
                    triangle_numbers[j*MaxDegree + k] = accumulator;
                    triangle_arcs[accumulator] = {j,carc.second};
                    accumulator++;
                }
            }
        }
        for (int j = 0; j < N; j++){
            uint16_t u = triangle_arcs[j].first;
            uint16_t v = triangle_arcs[j].second;
            uint16_t w = G.next(u,v);
            arc_t arc_a = G.canon_arc(v,u);
            arc_t arc_b = G.canon_arc(w,v);
            arc_t arc_c = G.canon_arc(u,w);
            G_out[i*N*3 + j*3 + 0] = triangle_numbers[arc_a.first*MaxDegree + G.dedge_ix(arc_a.first, arc_a.second)];
            G_out[i*N*3 + j*3 + 1] = triangle_numbers[arc_b.first*MaxDegree + G.dedge_ix(arc_b.first, arc_b.second)];
            G_out[i*N*3 + j*3 + 2] = triangle_numbers[arc_c.first*MaxDegree + G.dedge_ix(arc_c.first, arc_c.second)];
        }

    }   
    }
         
} 