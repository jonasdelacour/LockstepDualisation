#include "dual.h"
#include "cu_array.h"
#include "util.h"
#include "fstream"
#include "iostream"
#include "filesystem"
#include "vector"
#include "chrono"
#include "numeric"
#include "cmath"
#include "launch_ctx.h"
using namespace std::chrono_literals;
using namespace std::chrono;

int main(int argc, char** argv) {
    int N = argc > 1 ? std::stoi(argv[1]) : 200;
    int N_graphs = argc > 2 ? std::stoi(argv[2]) : 1000;
    int N_runs = argc > 3 ? std::stoi(argv[3]) : 100;
    int N_warmup = argc > 4 ? std::stoi(argv[4]) : 5;
    int version = argc > 5 ? std::stoi(argv[5]) : 0;
    std::string filename = argc > 6 ? argv[6] : "single_gpu.csv";
    
    std::ifstream file_check(filename);
    std::ofstream file(filename, std::ios_base::app);
    //If the file is empty, write the header.
    if(file_check.peek() == std::ifstream::traits_type::eof()) file << "N,BS,T,TSD,TD,TDSD\n"; 


    if(N==22 || N%2==1 || N<20 || N>200){
        std::cout << "N must be even and between 20 and 200 and not equal to 22." << std::endl;
        return 1;
    }
    int Nf = N/2 + 2;


    CuArray<uint16_t> graphs(N_graphs*Nf*6);               //Allocate unified memory to store the batch.
    CuArray<uint8_t> degrees(N_graphs*Nf);
    CuArray<uint16_t> out_graphs(N_graphs*N*3);               //Allocate unified memory to store the batch.

    fill(graphs, degrees, Nf, N_graphs);
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

    remove_outliers(times);

    file
        << N << ","
        << N_graphs << "," 
        << mean(times)/N_graphs << "," 
        << stddev(times)/N_graphs << "," 
        << mean(tdiffs)/N_graphs << ","
        << stddev(tdiffs)/N_graphs << "\n";
    
    std::cout << "Mean Time per Graph: " << mean(times) / N_graphs << " +/- " << stddev(times) / N_graphs << " ns" << std::endl;
    return 0;
}
