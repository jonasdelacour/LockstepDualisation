#include "fullerenes/polyhedron.hh"
#include "fullerenes/isomerdb.hh"
#include <chrono>
#include <fstream>
#include <stdio.h>
#include "random"
#include "numeric"
#define SYCLBATCH
#include "util.h"
#include "sycl_kernels.h"
#include "fullerenes/polyhedron.hh"

int main(int argc, char** argv) {
    // Make user be explicit about which device we want to test, to avoid surprises.
    if(argc < 2 || (string(argv[1]) != "cpu" && string(argv[1]) != "gpu")){
      fprintf(stderr, "Syntax: %s <cpu|gpu> [N_start:20] [N_end:200]\n",argv[0]);
      return -1;
    }
    
    std::string device_type = argv[1];
    size_t start_range = argc > 2 ? std::stoi(argv[2]) : 20;
    size_t end_range = argc > 3 ? std::stoi(argv[3]) : 200;

    auto selector =  device_type == "cpu" ? sycl::cpu_selector_v : sycl::gpu_selector_v;
    auto Q = sycl::queue(selector);

    printf("Validating SYCL implementation for %s device: %s.\n",
	   argv[1], Q.get_device().get_info<sycl::info::device::name>().c_str());
    size_t total_validated = 0;
    for (size_t N = start_range; N <= end_range; N+=2) {
        if (N == 22) continue;
        int Nf = N/2 + 2;
        int N_graphs = min<size_t>(10000, IsomerDB::number_isomers(N));

	printf("Validating dualization of %d C%ld-isomers against reference results.\n",N_graphs,N);
	
	
        IsomerBatch<float,uint16_t> batch(N, N_graphs);
        std::vector<FullereneDual> baseline_duals(N_graphs);
        for (int i = 0; i < N_graphs; ++i) {
            baseline_duals[i].neighbours = neighbours_t(Nf,std::vector<int>(6));
            baseline_duals[i].N = Nf;
        }
        std::vector<PlanarGraph> cubic_truth(N_graphs);

        fill(batch);
        //This scope is necessary to ensure that the host accessors are destroyed before the dualisation, otherwise dualise will block execution forever.
        {
            sycl::host_accessor<uint16_t,1> h_cubic_neighbours(batch.cubic_neighbours);
            sycl::host_accessor<uint16_t,1> h_dual_neighbours(batch.dual_neighbours);
            for (size_t i = 0; i < N_graphs; i++)
                for (size_t j = 0; j < Nf; j++){
                    baseline_duals[i].neighbours[j].clear();
                    for (size_t k = 0; k < 6; k++){
                        if (h_dual_neighbours[i*Nf*6 + j*6 + k] == UINT16_MAX) continue;
                        baseline_duals[i].neighbours[j].push_back(h_dual_neighbours[i*Nf*6 + j*6 + k]);
                    }
                }
            for (size_t i = 0; i < N_graphs; i++){
                baseline_duals[i].update();
                cubic_truth[i] = baseline_duals[i].dual_graph();
            }
        }
        



        //Check that the results are correct.
        //Create lambda function to check if two graphs are equal.
        auto check_graphs = [&](){
            sycl::host_accessor<uint16_t,1> h_cubic_neighbours(batch.cubic_neighbours);
            sycl::host_accessor<uint16_t,1> h_dual_neighbours(batch.dual_neighbours);
            for (size_t i = 0; i < N_graphs; i++){
            for(size_t j = 0; j < N; j++){
                if (h_cubic_neighbours[i*N*3 + j*3 + 0] != cubic_truth[i].neighbours[j][0] ||
                    h_cubic_neighbours[i*N*3 + j*3 + 1] != cubic_truth[i].neighbours[j][1] ||
                    h_cubic_neighbours[i*N*3 + j*3 + 2] != cubic_truth[i].neighbours[j][2]){
                    std::cout << "Error at " << i << " " << j << std::endl;
                    std::cout << "Expected " << cubic_truth[i].neighbours[j][0] << " " << cubic_truth[i].neighbours[j][1] << " " << cubic_truth[i].neighbours[j][2] << std::endl;
                    std::cout << "Got " << h_cubic_neighbours[i*N*3 + j*3 + 0] << " " << h_cubic_neighbours[i*N*3 + j*3 + 1] << " " << h_cubic_neighbours[i*N*3 + j*3 + 2] << std::endl;
                    return 1;
                }
            }
            }
            return 0;
        };

        dualise_sycl_v0<6>(Q, batch);
        if (check_graphs()) return 1;
        dualise_sycl_v1<6>(Q, batch);
        if (check_graphs()) return 1;

	total_validated += N_graphs;
    }
    printf("Success! All %ld dualized graphs were identical to reference results.\n",total_validated);
    return 0;
}
