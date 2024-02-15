#define SYCLBATCH
#include "../../include/isomer_batch.h"
#include <iostream>
using namespace sycl;

template <typename T, typename K>
IsomerBatch<T,K>::IsomerBatch(size_t n_atoms, size_t n_isomers) :    n_atoms           (n_atoms), m_capacity(n_isomers),
                                                                n_faces           (n_atoms / 2 + 2),
                                                                X                 (range<1>(n_isomers * n_atoms)), 
                                                                xys               (range<1>(n_isomers * n_atoms)), 
                                                                cubic_neighbours  (range<1>(n_isomers * n_atoms * 3)), 
                                                                dual_neighbours   (range<1>(6 * n_isomers * (n_atoms / 2 + 2))), 
                                                                face_degrees      (range<1>((n_atoms / 2 + 2) * n_isomers)), 
                                                                IDs               (range<1>(n_isomers)), 
                                                                iterations        (range<1>(n_isomers)), 
                                                                statuses          (range<1>(n_isomers))
{
    
    sycl::host_accessor X_acc(X, no_init);
    sycl::host_accessor xys_acc(xys, no_init);
    sycl::host_accessor cubic_neighbours_acc(cubic_neighbours, no_init);
    sycl::host_accessor dual_neighbours_acc(dual_neighbours, no_init);
    sycl::host_accessor face_degrees_acc(face_degrees, no_init);
    sycl::host_accessor IDs_acc(IDs, no_init);
    sycl::host_accessor iterations_acc(iterations, no_init);
    sycl::host_accessor statuses_acc(statuses, no_init);

    for (size_t i = 0; i < n_isomers; i++)
    {
        for (size_t j = 0; j < n_atoms; j++)
        {
            X_acc[i * n_atoms + j] = coord3d{0.0, 0.0, 0.0};
            xys_acc[i * n_atoms + j] = coord2d{0.0, 0.0};
            cubic_neighbours_acc[i * n_atoms * 3 + j * 3 + 0] = std::numeric_limits<K>::max();
            cubic_neighbours_acc[i * n_atoms * 3 + j * 3 + 1] = std::numeric_limits<K>::max();
            cubic_neighbours_acc[i * n_atoms * 3 + j * 3 + 2] = std::numeric_limits<K>::max();
        }
        for (size_t j = 0; j < 6 * n_faces; j++)
        {
            dual_neighbours_acc[i * 6 * n_faces + j] = std::numeric_limits<K>::max();
        }
        for (size_t j = 0; j < n_faces; j++)
        {
            face_degrees_acc[i * n_faces + j] = std::numeric_limits<K>::max();
        }
        IDs_acc[i] = std::numeric_limits<size_t>::max();
        iterations_acc[i] = 0;
        statuses_acc[i] = IsomerStatus::EMPTY;
    }
}   

template <typename T, typename K>
IsomerBatch<T,K>::IsomerBatch() : IsomerBatch(0, 0) {}

template <typename T, typename K>
void copy(IsomerBatch<T,K>& dest, IsomerBatch<T,K>& other){
    sycl::host_accessor X_acc(dest.X, write_only);
    sycl::host_accessor xys_acc(dest.xys, write_only);
    sycl::host_accessor cubic_neighbours_acc(dest.cubic_neighbours, write_only);
    sycl::host_accessor dual_neighbours_acc(dest.dual_neighbours, write_only);
    sycl::host_accessor face_degrees_acc(dest.face_degrees, write_only);
    sycl::host_accessor IDs_acc(dest.IDs, write_only);
    sycl::host_accessor iterations_acc(dest.iterations, write_only);
    sycl::host_accessor statuses_acc(dest.statuses, write_only);
    
    sycl::host_accessor other_X_acc(other.X, read_only);
    sycl::host_accessor other_xys_acc(other.xys, read_only);
    sycl::host_accessor other_cubic_neighbours_acc(other.cubic_neighbours, read_only);
    sycl::host_accessor other_dual_neighbours_acc(other.dual_neighbours, read_only);
    sycl::host_accessor other_face_degrees_acc(other.face_degrees, read_only);
    sycl::host_accessor other_IDs_acc(other.IDs, read_only);
    sycl::host_accessor other_iterations_acc(other.iterations, read_only);
    sycl::host_accessor other_statuses_acc(other.statuses, read_only);
    auto N = dest.n_atoms;
    auto Nf = dest.n_faces;

    for (size_t i = 0; i < dest.m_capacity; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            X_acc[i * N + j] = other_X_acc[i * N + j];
            xys_acc[i * N + j] = other_xys_acc[i * N + j];
            for (size_t k = 0; k < 3; k++)
            {
                cubic_neighbours_acc[i * N * 3 + j * 3 + k] = other_cubic_neighbours_acc[i * N * 3 + j * 3 + k];
            }
        }
        for (size_t j = 0; j < 6 * Nf; j++)
        {
            dual_neighbours_acc[i * 6 * Nf + j] = other_dual_neighbours_acc[i * 6 * Nf + j];
        }
        for (size_t j = 0; j < Nf; j++)
        {
            face_degrees_acc[i * Nf + j] = other_face_degrees_acc[i * Nf + j];
        }
        IDs_acc[i] = other_IDs_acc[i];
        iterations_acc[i] = other_iterations_acc[i];
        statuses_acc[i] = other_statuses_acc[i];
    }

}

template IsomerBatch<float, uint16_t>::IsomerBatch(size_t n_atom, size_t n_isomers);
template IsomerBatch<float, uint16_t>::IsomerBatch();

template void copy<float,uint16_t>(IsomerBatch<float, uint16_t>& dest, IsomerBatch<float, uint16_t>& other);