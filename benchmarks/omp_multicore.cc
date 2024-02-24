#include "cpp_kernels.h"
#include "util.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std::chrono;
int main(int argc, char** argv){
  const char *version_ptr = getenv("DUALIZE_VERSION");
  int version = version_ptr? strtol(version_ptr,0,0) : 0;
  
  int N = argc > 1 ? std::stoi(argv[1]) : 200;
  int N_graphs = argc > 2 ? std::stoi(argv[2]) : 10000;
  int N_runs = argc > 3 ? std::stoi(argv[3]) : 10;
  int N_warmup = argc > 4 ? std::stoi(argv[4]) : 5;
  version      = argc > 5 ? std::stoi(argv[5]) : version;
  
  std::string filename = argc > 6 ? argv[6] : "omp_multicore.csv";
  std::cout << "Dualising " << N_graphs << " triangulation graphs, each with " << N
              << " triangles, repeated " << N_runs << " times and with " << N_warmup
              << " warmup runs." << std::endl;

    if(N==22 || N%2==1 || N<20 || N>200){
        std::cout << "N must be even and between 20 and 200 and not equal to 22." << std::endl;
        return 1;
    }

    std::ifstream file_check(filename);
    std::ofstream file(filename, std::ios_base::app);
    //If the file is empty, write the header.
    if(file_check.peek() == std::ifstream::traits_type::eof()) file << "N,BS,T,TSD,TD,TDSD\n";

    int Nf = N/2 + 2;
    IsomerBatch<float,uint16_t> batch(N, N_graphs);

    fill(batch);

    std::vector<double> times(N_runs); //Times in nanoseconds.
    std::chrono::steady_clock::time_point start;
    for (size_t i = 0; i < N_runs + N_warmup; i++)
    {
        if(i >= N_warmup) start = steady_clock::now();

        switch (version)
        {
        case 0: //Shared-memory parallel version.
            dualise_omp_shared<6>(batch);
            break;
        case 1: //Task parallel version.
            dualise_omp_task<6>(batch);
            break;
        default:
            break;
        }

        if(i >= N_warmup) times[i-N_warmup] = duration<double,std::nano>(steady_clock::now() - start).count();
    }

    remove_outliers(times,1);
    remove_outliers(times,2);
    remove_outliers(times,3);

    file
        << N << ","
        << N_graphs << ","
        << mean(times)/N_graphs << ","
        << stddev(times)/N_graphs << ",,\n";

    std::cout << "Mean Time per Graph: " << mean(times) / N_graphs << " +/- " << stddev(times) / N_graphs << " ns" << std::endl;
    return 0;
}
