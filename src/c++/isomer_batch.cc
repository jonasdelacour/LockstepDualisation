#define CPPBATCH
#include "../../include/isomer_batch.h"
#include <limits>
#include <iostream>
#include <stdint.h>

template <typename T, typename K>
IsomerBatch<T,K>::IsomerBatch(size_t n_atoms, size_t n_isomers){
    this->n_atoms = n_atoms;
    this->n_faces = n_atoms / 2 + 2;
    this->m_capacity = n_isomers;
    this->X.resize(n_isomers * n_atoms, coord3d{0.0, 0.0, 0.0});
    this->xys.resize(n_isomers * n_atoms, coord2d{0.0, 0.0});
    this->cubic_neighbours.resize(n_isomers * n_atoms * 3, std::numeric_limits<K>::max());
    this->dual_neighbours.resize(6 * n_isomers * n_faces, std::numeric_limits<K>::max());
    this->face_degrees.resize(n_isomers * n_faces, std::numeric_limits<K>::max());
    this->IDs.resize(n_isomers, std::numeric_limits<size_t>::max());
    this->iterations.resize(n_isomers, 0);
    this->statuses.resize(n_isomers, IsomerStatus::EMPTY);
}

template <typename T, typename K>
IsomerBatch<T,K>::IsomerBatch() : IsomerBatch(0, 0) {}

template IsomerBatch<float, uint16_t>::IsomerBatch(size_t n_atoms, size_t n_isomers);
template IsomerBatch<float, uint16_t>::IsomerBatch();