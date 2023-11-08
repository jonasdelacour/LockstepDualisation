#define CUDABATCH
#include "../../include/isomer_batch.h"

template <typename T, typename K>
IsomerBatch<T,K>::IsomerBatch(size_t n_atoms, size_t n_isomers){
    this->n_atoms = n_atoms;
    this->m_capacity = n_isomers;
    this->n_faces = n_atoms/2 + 2;

    cudaMallocManaged((void**)&this->X,                 3 * n_isomers * n_atoms *    sizeof(T));
    cudaMallocManaged((void**)&this->xys,               2 * n_isomers * n_atoms *  sizeof(T));
    cudaMallocManaged((void**)&this->cubic_neighbours,  3 * n_isomers * n_atoms * sizeof(K));
    cudaMallocManaged((void**)&this->dual_neighbours,   6 * n_isomers * n_atoms *  sizeof(K));
    cudaMallocManaged((void**)&this->face_degrees,      n_isomers * n_faces * sizeof(K));
    cudaMallocManaged((void**)&this->IDs,               n_isomers * sizeof(K));
    cudaMallocManaged((void**)&this->iterations,        n_isomers * sizeof(K));
    cudaMallocManaged((void**)&this->statuses,          n_isomers * sizeof(IsomerStatus));
    
    cudaMemset(this->X, 0, 3 * n_isomers * n_atoms *    sizeof(T));
    cudaMemset(this->xys, 0, 2 * n_isomers * n_atoms *  sizeof(T));
    cudaMemset(this->cubic_neighbours, 0,  3 * n_isomers * n_atoms * sizeof(K));
    cudaMemset(this->dual_neighbours, 0,   6 * n_isomers * n_atoms *  sizeof(K));
    cudaMemset(this->face_degrees, 0,      n_isomers * n_faces * sizeof(K));
    cudaMemset(this->IDs, 0,               n_isomers * sizeof(K));
    cudaMemset(this->iterations, 0,        n_isomers * sizeof(K));
    cudaMemset(this->statuses, 0,          n_isomers * sizeof(IsomerStatus));
    this->allocated = true;
}

template <typename T, typename K>
IsomerBatch<T,K>::IsomerBatch(){
    this->allocated = false;
}


template IsomerBatch<float, uint16_t>::IsomerBatch(size_t n_atom, size_t n_isomers);
template IsomerBatch<float, uint16_t>::IsomerBatch();