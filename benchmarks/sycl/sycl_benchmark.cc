#include "fstream"
#include "iostream"
#include "vector"
#include "chrono"
#include "numeric"
#include "cmath"
#include "algorithm"
#define SYCLBATCH
#include "util.h"
#include "sycl_kernels.h"

using namespace std::chrono_literals;
using namespace std::chrono;

int main(int argc, char** argv) {
    int N = argc > 1 ? std::stoi(argv[1]) : 200;
    int N_graphs = argc > 2 ? std::stoi(argv[2]) : 1000000;
    int N_runs = argc > 3 ? std::stoi(argv[3]) : 10;
    int N_warmup = argc > 4 ? std::stoi(argv[4]) : 1;
    int version = argc > 5 ? std::stoi(argv[5]) : 0;
    std::string filename = argc > 6 ? argv[6] : "multi_gpu.csv";
    std::cout << "Dualising " << N_graphs << " triangulation graphs, each with " << N
              << " triangles, repeated " << N_runs << " times and with " << N_warmup
              << " warmup runs." << std::endl;

    int N_d = 0;
    std::vector<sycl::queue> Qs;
    for (auto platform : sycl::platform::get_platforms())
    {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;

        for (auto device : platform.get_devices())
        {
            std::cout << "\tDevice: "
                      << device.get_info<sycl::info::device::name>()
                      << std::endl;
            if (device.is_gpu()){
                N_d++;
                Qs.push_back(sycl::queue(device));
            }
        }
    }

    std::cout << "Found " << N_d << " GPUs." << std::endl;
    if(N==22 || N%2==1 || N<20 || N>200){
        std::cout << "N must be even and between 20 and 200 and not equal to 22." << std::endl;
        return 1;
    }

    std::ifstream file_check(filename);
    std::ofstream file(filename, std::ios_base::app);
    //If the file is empty, write the header.
    if(file_check.peek() == std::ifstream::traits_type::eof()) file << "N,BS,T,TSD,TD,TDSD\n";

    int Nf = N/2 + 2;

    //std::vector<IsomerBatch<float,uint16_t>> batches= {IsomerBatch<float,uint16_t>(N, N_graphs/N_d + N_graphs%N_d), IsomerBatch<float,uint16_t>(N, N_graphs/N_d + N_graphs%N_d)};

    std::vector<IsomerBatch<float,uint16_t>> batches; for(int i = 0; i < N_d; i++) batches.push_back(IsomerBatch<float,uint16_t>(N, N_graphs/N_d + N_graphs%N_d));

    for(int i = 0; i < N_d; i++) fill(batches[i]);


    std::vector<double> times(N_runs); //Times in nanoseconds.
    std::vector<double> tdiffs(N_runs); //Timing differences in nanoseconds.
    for (size_t i = 0; i < (N_runs + N_warmup); i++)
    {
        if(i >= N_warmup) {
            auto start = steady_clock::now();
            auto start_times = std::vector<steady_clock::time_point>(N_d);
            for(int j = 0; j < N_d; j++) start_times[j] = steady_clock::now();

            switch (version)
            {
            case 0:
                for(int j = 0; j < N_d; j++) dualise_sycl_v0<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);   //Dualise the batch.
                break;
            case 1:
                //for(int j = 0; j < N_d; j++) {dualise_sycl_v1<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);}    //Dualise the batch.
                break;
            default:
                break;
            }
            std::vector<double> t_gpus(N_d);
            for(int j = 0; j < N_d; j++) Qs[j].wait();
            for(int j = 0; j < N_d; j++) t_gpus[j] =  duration<double,std::nano>(steady_clock::now() - start_times[j]).count();
            times[i-N_warmup] = *std::max_element(t_gpus.begin(), t_gpus.end());
            auto end = steady_clock::now();
            tdiffs[i-N_warmup] = std::abs(duration<double,std::nano>(end - start).count()/N_graphs - duration<double,std::nano>(times[i-N_warmup]).count()/N_graphs);
        } else {
            switch (version)
            {
            case 0:
                for(int j = 0; j < N_d; j++) dualise_sycl_v0<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);    //Dualise the batch.
                break;
            case 1:
                //for(int j = 0; j < N_d; j++) dualise_sycl_v1<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);    //Dualise the batch.
                break;
            default:
                break;
            }
        }
    }

    //remove_outliers(times);

    //file
    //    << N << ","
    //    << N_graphs << ","
    //    << mean(times)/N_graphs << ","
    //    << stddev(times)/N_graphs << ","
    //    << mean(tdiffs)/N_graphs << ","
    //    << stddev(tdiffs)/N_graphs << "\n";
    std::cout << "Mean Time per Graph: " << mean(times) / N_graphs << " +/- " << stddev(times) / N_graphs << " ns\n";
    return 0;
}
