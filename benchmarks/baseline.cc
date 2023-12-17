#include <fullerenes/polyhedron.hh>
#include <fullerenes/isomerdb.hh>
#include <fullerenes/buckygen-wrapper.hh>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <stdio.h>
#include "random"
#include "numeric"
#define CPPBATCH
#include "util.h"

using namespace chrono;

int main(int argc, char** argv) {
    int N =        argc > 1 ? std::stoi(argv[1]) : 200;
    int N_graphs = argc > 2 ? std::stoi(argv[2]) : 1000;
    int N_runs =   argc > 3 ? std::stoi(argv[3]) : 10;
    int N_warmup = argc > 4 ? std::stoi(argv[4]) : 1;

    std::string filename = argc > 6 ? argv[6]    : "baseline.csv";
    printf("Dualising %d triangulation graphs, each with %d triangles, repeated %d times and with %d warmup runs.\n",
	   N_graphs, N, N_runs, N_warmup);

    std::ifstream file_check(filename);
    std::ofstream file(filename, std::ios_base::app);
    //If the file is empty, write the header.
    if(file_check.peek() == std::ifstream::traits_type::eof()) file << "N,BS,T,TSD,TD,TDSD\n";
    if(N==22 || N%2==1 || N<20 || N>200) {
        std::cout  << "N must be even and between 20 and 200 and not equal to 22.\n";
        return -1;
    }
    
    // 1. Fill buffer with graphs
    int Nf = N/2+2; // Number of vertices in triangulation     
    vector<FullereneDual> in_graphs(N_graphs,Triangulation(Nf));
    auto BuckyQ = BuckyGen::start(N);
    bool more_isomers = false;
    int isomer_ix     = 0;
    while((isomer_ix<N_graphs) && (more_isomers = BuckyGen::next_fullerene(BuckyQ,in_graphs[isomer_ix]))) { isomer_ix++; }
    if(!more_isomers)		// If isomer space is too small, repeat to fill buffer.
      for(int i=isomer_ix-1;i<N_graphs;i++) in_graphs[i] = in_graphs[i%isomer_ix];

    // 2. Dualize them all and time
    vector<double> times(N_runs,0);
    
    for(int i=0;i<N_warmup+N_runs;i++){
      auto T0 = steady_clock::now();      
      for(int j=0;j<N_graphs;j++){
	in_graphs[isomer_ix].update();              // Computes triangles/faces -- part of dualization algorithm we want to time
	PlanarGraph G = in_graphs[j].dual_graph();  // Computes the dual graph given the faces.
      }
      auto T1 = steady_clock::now();
      if(i >= N_warmup) times[i-N_warmup] = duration<double,std::nano>(T1-T0).count();
    }

    //Removes data points that are more than 3 standard deviations away from the mean. (Can be adjusted)
    remove_outliers(times);

    file
        << N << ","
        << N_graphs << ","
        << mean(times) / N_graphs << ","
        << stddev(times) / N_graphs << ",,\n";
    std::cout << "Mean Time per Graph: " << mean(times) / N_graphs << " +/- " << stddev(times) / N_graphs << " ns" << std::endl;
    return 0;
}
