#include "fullerenes/polyhedron.hh"
#include "fullerenes/isomerdb.hh"
#include <chrono>
#include <fstream>
#include <unistd.h>
#include "random"
#include "numeric"
#define CPPBATCH
#include "util.h"

using namespace chrono;

int main(int argc, char** argv) {
    int N =        argc > 1 ? std::stoi(argv[1]) : 200;
    int N_graphs = argc > 2 ? std::stoi(argv[2]) : 1000;
    int N_runs =   argc > 3 ? std::stoi(argv[3]) : 10;
    int N_warmup = argc > 4 ? std::stoi(argv[4]) : 5;
    int version =  argc > 5 ? std::stoi(argv[5]) : 0;
    std::string filename = argc > 6 ? argv[6]    : "baseline.csv";
    std::cout << "Dualising " << N_graphs << " triangulation graphs, each with " << N
              << " triangles, repeated " << N_runs << " times and with " << N_warmup
              << " warmup runs." << std::endl;

    std::ifstream file_check(filename);
    std::ofstream file(filename, std::ios_base::app);
    //If the file is empty, write the header.
    if(file_check.peek() == std::ifstream::traits_type::eof()) file << "N,BS,T,TSD,TD,TDSD\n";
    if(N==22 || N%2==1 || N<20 || N>200){
        std::cout
            << "N must be even and between 20 and 200 and not equal to 22."
            << std::endl;
        return 1;
    }

    auto sample_size = min<size_t>(200,IsomerDB::number_isomers(N));
    int Nf = N/2 + 2;
    std::vector<node_t> in_graphs(N_graphs*Nf*6);
    std::vector<uint8_t> in_degrees(N_graphs*Nf);
    std::vector<node_t> out_graphs(N_graphs*N*3);
    FullereneDual G;
    G.neighbours = neighbours_t(Nf, std::vector<node_t>(6));
    G.N = Nf;

    auto path = cwd() +"/isomerspace_samples/dual_layout_" + to_string(N) + "_seed_42";
    ifstream isomer_sample(path,std::ios::binary);
    auto fsize = filesize(isomer_sample);

    std::vector<uint16_t> input_buffer(fsize/sizeof(uint16_t));
    auto available_samples = fsize / (Nf*6*sizeof(uint16_t));
    isomer_sample.read(
        reinterpret_cast<char*>(input_buffer.data()),
        Nf*6*sizeof(uint16_t)*available_samples);


    std::vector<double> times(N_runs); //Times in nanoseconds.
    for (size_t i = 0; i < N_runs + N_warmup; i++) {
        chrono::_V2::steady_clock::time_point start = steady_clock::now();

        for (int j = 0; j < N_graphs; ++j) {
            for (size_t k = 0; k < Nf; k++) {
                G.neighbours[k].clear();
                for (size_t ii = 0; ii < 6; ii++) {
                    auto u = input_buffer[(j%available_samples)*Nf*6 + k*6 +ii];
                    if(u != UINT16_MAX) G.neighbours[k].push_back(u);
                }
            }

            G.update();
            PlanarGraph pG = G.dual_graph();
        }
        if(i >= N_warmup)
            times[i-N_warmup] = duration<double,std::nano>(
                steady_clock::now() - start).count();
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
