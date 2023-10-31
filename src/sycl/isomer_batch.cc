#define SYCLBATCH
#include "../../include/isomer_batch.h"
using namespace sycl;

template <typename T, typename K>
IsomerBatch<T,K>::IsomerBatch(size_t n_atoms, size_t n_isomers) :    n_atoms           (n_atoms), m_capacity(n_isomers),
                                                                n_faces           (n_atoms / 2 + 2),
                                                                X                 (range<1>(n_isomers * n_atoms)), 
                                                                xys               (range<1>(n_isomers * n_atoms)), 
                                                                cubic_neighbours  (range<1>(n_isomers * n_atoms * 3)), 
                                                                dual_neighbours   (range<1>(6 * n_isomers * (n_atoms / 2 + 2))), 
                                                                face_degrees      (range<1>((n_atoms / 2 + 2) * 1)), 
                                                                IDs               (range<1>(n_isomers)), 
                                                                iterations        (range<1>(n_isomers)), 
                                                                statuses          (range<1>(n_isomers))
{
    X                 = buffer<coord3d, 1>( range<1>(n_isomers * n_atoms));
    xys               = buffer<coord2d, 1>( range<1>(n_isomers * n_atoms));
    cubic_neighbours  = buffer<K, 1>( range<1>(n_isomers * n_atoms * 3));
    dual_neighbours   = buffer<K, 1>( range<1>(6 * n_isomers * n_faces));
    face_degrees      = buffer<K, 1>( range<1>(n_isomers * n_faces));
    IDs               = buffer<size_t, 1>( range<1>(n_isomers));
    iterations        = buffer<size_t, 1>( range<1>(n_isomers));
    statuses          = buffer<IsomerStatus, 1>( range<1>(n_isomers));
    
    host_accessor X_acc(X, no_init);
    host_accessor xys_acc(xys, no_init);
    host_accessor cubic_neighbours_acc(cubic_neighbours, no_init);
    host_accessor dual_neighbours_acc(dual_neighbours, no_init);
    host_accessor face_degrees_acc(face_degrees, no_init);
    host_accessor IDs_acc(IDs, no_init);
    host_accessor iterations_acc(iterations, no_init);
    host_accessor statuses_acc(statuses, no_init);

        


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


template IsomerBatch<float, uint16_t>::IsomerBatch(size_t n_atom, size_t n_isomers);
