#include <vector>
#include "numeric"
#include "cmath"
template <typename T>
T mean(const std::vector<T>& v);

template<typename T>
T stddev(const std::vector<T>& data);

template <typename T, typename U>
void fill(T& G_in, U& degrees, const int Nf, const int N_graphs);
