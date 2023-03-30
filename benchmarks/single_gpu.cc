#include "../include/dual.h"
#include "fstream"
#include "iostream"
#include "../include/cu_array.h"
#include "filesystem"
#include "vector"
#include "chrono"
#include "numeric"
#include "cmath"
using namespace std::chrono_literals;
using namespace std::chrono;

template <typename T>
T mean(std::vector<T> v) {
    T sum = 0;
    return std::reduce(v.begin(), v.end(), sum) / v.size();
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

int main(int argc, char** argv) {
    int N = argc > 1 ? std::stoi(argv[1]) : 200;
    int N_graphs = argc > 2 ? std::stoi(argv[2]) : 1000;
    int N_runs = argc > 3 ? std::stoi(argv[3]) : 100;
    int N_warmup = argc > 4 ? std::stoi(argv[4]) : 5;
    int version = argc > 5 ? std::stoi(argv[5]) : 0;
    if(N==22 || N%2==1 || N<20 || N>200){
        std::cout << "N must be even and between 20 and 200 and not equal to 22." << std::endl;
        return 1;
    }
    int Nf = N/2 + 2;
    const std::string path = "../isomerspace_samples/dual_layout_" + std::to_string(N) + "_seed_42";
    std::ifstream samples(path, std::ios::binary);        //Open the file containing the samples.
    int fsize = std::filesystem::file_size(path);         //Get the size of the file in bytes.
    int n_samples = fsize / (Nf * 6 * sizeof(uint16_t));   //All the graphs are fullerene graphs stored in 16bit unsigned integers.

    std::vector<uint16_t> in_buffer(n_samples * Nf * 6);   //Allocate a buffer to store all the samples.
    samples.read((char*)in_buffer.data(), n_samples*Nf*6*sizeof(uint16_t));         //Read all the samples into the buffer.

    CuArray<uint16_t> graphs(N_graphs*Nf*6);               //Allocate unified memory to store the batch.
    CuArray<uint8_t> degrees(N_graphs*Nf);
    CuArray<uint16_t> out_graphs(N_graphs*N*3);               //Allocate unified memory to store the batch.
    for(int i = 0; i < N_graphs; i++) {                  //Copy the first N_graphs samples into the batch.
        for(int j = 0; j < Nf; j++) {
        for(int k = 0; k < 6; k++) {
            graphs[i*Nf*6 + j*6 + k] = in_buffer[(i%n_samples)*Nf*6 + j*6 + k];
            if(k==5) degrees[i*Nf + j] = in_buffer[(i%n_samples)*Nf*6 + j*6 + k] == UINT16_MAX ? 5 : 6;
        }
        }
    }
    LaunchCtx ctx(0);

    graphs.to_device(0);                                  //Copy the batch to the device.
    degrees.to_device(0);                                 //Copy the degrees to the device.
    out_graphs.to_device(0);                                 //Copy the degrees to the device.

    std::vector<double> times(N_runs); //Times in nanoseconds.
    std::vector<double> tdiffs(N_runs); //Timing differences in nanoseconds.

    for (size_t i = 0; i < (N_runs + N_warmup); i++)
    {
        auto start = steady_clock::now();
        ctx.start_timer();
        switch (version)
        {
        case 0:
            dualise_V0<6>(graphs, degrees, out_graphs, Nf, N_graphs, ctx, LaunchPolicy::ASYNC);    //Dualise the batch.
            break;
        case 1:
            dualise_V1<6>(graphs, degrees, out_graphs, Nf, N_graphs, ctx, LaunchPolicy::ASYNC);    //Dualise the batch.
            break;
        default:
            break;
        }
        if(i >= N_warmup) times[i-N_warmup] = duration<double,std::nano>(ctx.stop_timer()).count();
        auto end = steady_clock::now(); 
        if(i >= N_warmup) tdiffs[i-N_warmup] = std::abs(duration<double,std::nano>(end - start).count()/N_graphs - duration<double,std::nano>(times[i-N_warmup]).count()/N_graphs);
    }
    
    std::cout << "N\t | Time\t | Time SD\t | Time Diff\t | Time Diff SD" << std::endl;
    std::cout << N << ", " << mean(times)/N_graphs << ", " << stddev(times)/N_graphs << ", " <<  mean(tdiffs)/N_graphs << ", " << stddev(tdiffs)/N_graphs << std::endl;
    
    return 0;
}