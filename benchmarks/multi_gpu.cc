#include "../include/dual.h"
#include "fstream"
#include "iostream"
#include "../include/cu_array.h"
#include "filesystem"
#include "vector"
#include "chrono"
#include "numeric"
#include "cmath"
#include "algorithm"
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
    int N_runs = argc > 3 ? std::stoi(argv[3]) : 1;
    int version = argc > 4 ? std::stoi(argv[4]) : 0;
    int N_warmup = argc > 5 ? std::stoi(argv[5]) : 1;
    int N_d = LaunchCtx::get_device_count();


    int Nf = N/2 + 2;
    const std::string path = "../isomerspace_samples/dual_layout_" + std::to_string(N) + "_seed_42";
    std::ifstream samples(path, std::ios::binary);        //Open the file containing the samples.
    int fsize = std::filesystem::file_size(path);         //Get the size of the file in bytes.
    int n_samples = fsize / (Nf * 6 * sizeof(uint16_t));   //All the graphs are fullerene graphs stored in 16bit unsigned integers.

    std::vector<uint16_t> in_buffer(n_samples * Nf * 6);   //Allocate a buffer to store all the samples.
    samples.read((char*)in_buffer.data(), n_samples*Nf*6*sizeof(uint16_t));         //Read all the samples into the buffer.

    std::vector<CuArray<uint16_t>> in_graphs(N_d); for(int i = 0; i < N_d; i++) {in_graphs[i]=CuArray<uint16_t>((N_graphs/N_d + N_graphs%N_d)*Nf*6);}
    std::vector<CuArray<uint8_t>> in_degrees(N_d); for(int i = 0; i < N_d; i++) {in_degrees[i]=CuArray<uint8_t>((N_graphs/N_d + N_graphs%N_d)*Nf);}
    std::vector<CuArray<uint16_t>> out_graphs(N_d); for(int i = 0; i < N_d; i++) {out_graphs[i]=CuArray<uint16_t>((N_graphs/N_d + N_graphs%N_d)*N*3);}
    std::vector<int> counts(N_d, N_graphs/N_d); counts[N_d-1] += N_graphs%N_d;
    for(int dev_id = 0; dev_id < N_d; dev_id++) {                  //Copy the first N_graphs samples into the batch.
        int idx = 0;
        for(int i = 0; i < counts[dev_id]; i++) {
        for(int j = 0; j < Nf; j++) {
        for(int k = 0; k < 6; k++) {
            in_graphs[dev_id][i*Nf*6 + j*6 + k] = in_buffer[(i%n_samples)*Nf*6 + j*6 + k];
            if(k==5) in_degrees[dev_id][i*Nf + j] = in_buffer[(i%n_samples)*Nf*6 + j*6 + k] == UINT16_MAX ? 5 : 6;
        }
        }
        }
    }

    
    std::vector<LaunchCtx> ctxs(N_d); for(int i = 0; i < N_d; i++) {ctxs[i] = LaunchCtx(i);}

    for(int i = 0; i < N_d; i++) {
        in_graphs[i].to_device(i);
        in_degrees[i].to_device(i);
        out_graphs[i].to_device(i);
    }
    
    std::vector<double> times(N_runs); //Times in nanoseconds.
    std::vector<double> tdiffs(N_runs); //Timing differences in nanoseconds.
    for (size_t i = 0; i < (N_runs + N_warmup); i++)
    {   
        if(i >= N_warmup) {
            auto start = steady_clock::now();
            for(int j = 0; j < N_d; j++) ctxs[j].start_timer();
        
            switch (version)
            {
            case 0:
                for(int j = 0; j < N_d; j++) {dualise_V0<6>(in_graphs[j], in_degrees[j], out_graphs[j], Nf, N_graphs/N_d, ctxs[j], LaunchPolicy::ASYNC);}
                break;
            case 1:
                for(int j = 0; j < N_d; j++) {dualise_V1<6>(in_graphs[j], in_degrees[j], out_graphs[j], Nf, N_graphs/N_d, ctxs[j], LaunchPolicy::ASYNC);}    //Dualise the batch.
                break;
            default:
                break;
            }
            std::vector<double> t_gpus(N_d);
        
            for(int j = 0; j < N_d; j++) t_gpus[j] =  duration<double,std::nano>(ctxs[j].stop_timer()).count();
            times[i-N_warmup] = *std::max_element(t_gpus.begin(), t_gpus.end());
            auto end = steady_clock::now(); 
            tdiffs[i-N_warmup] = std::abs(duration<double,std::nano>(end - start).count()/N_graphs - duration<double,std::nano>(times[i-N_warmup]).count()/N_graphs);
        } else {
            switch (version)
            {
            case 0:
                for(int j = 0; j < N_d; j++) dualise_V0<6>(in_graphs[j], in_degrees[j], out_graphs[j], Nf, N_graphs/N_d, ctxs[j], LaunchPolicy::ASYNC);    //Dualise the batch.
                break;
            case 1:
                for(int j = 0; j < N_d; j++) dualise_V1<6>(in_graphs[j], in_degrees[j], out_graphs[j], Nf, N_graphs/N_d, ctxs[j], LaunchPolicy::ASYNC);    //Dualise the batch.
                break;
            default:
                break;
            }
        }
    }
    

    std::cout << "Time per Graph: " << mean(times)/N_graphs << "+/- " << stddev(times)/N_graphs << " ns" << std::endl;
    std::cout << "Timing Difference: " << mean(tdiffs)/N_graphs << "+/- " << stddev(tdiffs)/N_graphs << " ns" << std::endl;
    
    return 0;
}