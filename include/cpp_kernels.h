#define CPPBATCH
#include "isomer_batch.h"
template <int MaxDegree, typename T, typename K>
void dualise_omp_shared (IsomerBatch<T,K>& B);

template <int MaxDegree, typename T, typename K>
void dualise_omp_task   (IsomerBatch<T,K>& B);