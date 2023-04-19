#include "util.h"
#include "dual.h"
#include "fstream"
#include "filesystem"
#include "iostream"
template void fill(CuArray<node_t>& G_in, CuArray<uint8_t>& degrees, const int Nf, const int N_graphs);

template <typename T, typename U>
void fill(T& G_in, U& degrees, const int Nf, const int N_graphs) {
    int N = (Nf - 2)*2;
    std::filesystem::path p = std::filesystem::current_path();

    const std::string path = p.string() + "/isomerspace_samples/dual_layout_" + std::to_string(N) + "_seed_42";
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
