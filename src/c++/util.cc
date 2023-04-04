#include "util.h"
#include "numeric"
#include "cmath"
#include "dual.h"
#include "filesystem"
#include "fstream"

template float mean(std::vector<float> const& v);
template double mean(std::vector<double> const& v);

template float stddev(std::vector<float> const& data);
template double stddev(std::vector<double> const& data);

template void fill(std::vector<node_t>& G_in, std::vector<uint8_t>& degrees, const int Nf, const int N_graphs);

template <typename T>
T mean(const std::vector<T>& v) {
    return std::reduce(v.begin(), v.end(), T(0)) / v.size();
}

template<typename T>
T stddev(const std::vector<T>& data)
{
    // Calculate the mean
    T sum = std::reduce(data.begin(), data.end(), T(0));
    T mean = sum / data.size();

    // Calculate the sum of squared differences from the mean
    T sum_of_squares = 0.0;
    for (const T& value : data)
    {
        T diff = value - mean;
        sum_of_squares += diff * diff;
    }

    // Calculate the variance and return the square root
    T variance = sum_of_squares / (data.size() - 1);
    return std::sqrt(variance);
}

template <typename T, typename U>
void fill(T& G_in, U& degrees, const int Nf, const int N_graphs) {
    int N = (Nf - 2)*2;
    const std::string path = "../isomerspace_samples/dual_layout_" + std::to_string(N) + "_seed_42";
    std::ifstream samples(path, std::ios::binary);        //Open the file containing the samples.
    int fsize = std::filesystem::file_size(path);         //Get the size of the file in bytes.
    int n_samples = fsize / (Nf * 6 * sizeof(uint16_t));   //All the graphs are fullerene graphs stored in 16bit unsigned integers.

    std::vector<uint16_t> in_buffer(n_samples * Nf * 6);   //Allocate a buffer to store all the samples.
    samples.read((char*)in_buffer.data(), n_samples*Nf*6*sizeof(uint16_t));         //Read all the samples into the buffer.

    for(int i = 0; i < N_graphs; i++) {                  //Copy the first N_graphs samples into the batch.
        for(int j = 0; j < Nf; j++) {
        for(int k = 0; k < 6; k++) {
            G_in[i*Nf*6 + j*6 + k] = in_buffer[(i%n_samples)*Nf*6 + j*6 + k];
            if(k==5) degrees[i*Nf + j] = in_buffer[(i%n_samples)*Nf*6 + j*6 + k] == UINT16_MAX ? 5 : 6;
        }
        }
    }
}