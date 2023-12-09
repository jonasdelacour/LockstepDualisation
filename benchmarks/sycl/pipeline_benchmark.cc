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
using namespace std;

int main(int argc, char** argv) {
    // Make user be explicit about which device we want to test, to avoid surprises.
    if(argc < 2 || (string(argv[1]) != "cpu" && string(argv[1]) != "gpu")){
      fprintf(stderr, "Syntax: %s <cpu|gpu> [N:200] [N_graphs:1000000] [N_runs:10] [N_warmup:1] [version:0] [filename:multi_gpu.csv]\n",argv[0]);
      return -1;
    }
    string device_type = argv[1];
    bool want_gpus = (device_type == "gpu");
      
    // Parameters for the benchmark run
    int N = argc > 2 ? stoi(argv[2]) : 200;
    int N_graphs = argc > 3 ? stoi(argv[3]) : 1000000;
    int N_runs = argc > 4 ? stoi(argv[4]) : 10;
    int N_warmup = argc > 5 ? stoi(argv[5]) : 1;
    int version = argc > 6 ? stoi(argv[6]) : 0;
    string filename = argc > 7 ? argv[7] : "multi_"+device_type+".csv";
    
    cout << "Dualising " << N_graphs << " triangulation graphs, each with " << N
	 << " triangles, repeated " << N_runs << " times and with " << N_warmup
	 << " warmup runs." << endl;

    // Get all appropriate devices
    int N_d = 0;
    vector<sycl::queue> Qs;
    for (auto platform : sycl::platform::get_platforms())
    {
        cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << endl;

        for (auto device : platform.get_devices())
        {
	  bool use_this_device = (want_gpus && device.is_gpu())  || (!want_gpus && device.is_cpu() &&
								     (N_d<1/* NB: Sometimes SYCL reports the same CPU twice, leading to death and pain. */));
	  
	  if (use_this_device){
	    N_d++;
	    Qs.push_back(sycl::queue(device));
	  }
	  printf("\t%s: %s has %d compute-units.\n",
		 use_this_device? "USING    ":"NOT USING",
		 device.get_info<sycl::info::device::name>().c_str(),
		 device.get_info<sycl::info::device::max_compute_units>());
        }
    }

    printf("Using %d %s-devices\n",N_d,device_type.c_str());

    if(N==22 || N%2==1 || N<20 || N>200){
        cout << "N must be even and between 20 and 200 and not equal to 22." << endl;
        return 1;
    }

    ifstream file_check(filename);
    ofstream file(filename, ios_base::app);
    //If the file is empty, write the header.
    if(file_check.peek() == ifstream::traits_type::eof()) file << "N,BS,T,TSD,TD,TDSD\n";

    int Nf = N/2 + 2;

    //vector<IsomerBatch<float,uint16_t>> batches= {IsomerBatch<float,uint16_t>(N, N_graphs/N_d + N_graphs%N_d), IsomerBatch<float,uint16_t>(N, N_graphs/N_d + N_graphs%N_d)};

    vector<IsomerBatch<float,uint16_t>> batches; for(int i = 0; i < N_d; i++) batches.push_back(IsomerBatch<float,uint16_t>(N, N_graphs/N_d + N_graphs%N_d));

    for(int i = 0; i < N_d; i++) fill(batches[i]);


    vector<double> times_fill(N_runs); //Times in nanoseconds.
    vector<double> times_memcpy(N_runs); //Times in nanoseconds.
    vector<double> times_dual(N_runs); //Times in nanoseconds.
    vector<double> times_tutte(N_runs); //Times in nanoseconds.
    vector<double> times_project(N_runs); //Times in nanoseconds.
    vector<double> times_opt(N_runs); //Times in nanoseconds.

    for (size_t i = 0; i < (N_runs + N_warmup); i++)
    {
        if(i >= N_warmup) {
            auto start = steady_clock::now();
            for (int j = 0; j < N_d; j++) fill(batches[j]);
            times_fill[i-N_warmup] =  duration<double,nano>(steady_clock::now() - start).count();
            for(int j = 0; j < N_d; j++) {nop_kernel(Qs[j], batches[j], LaunchPolicy::SYNC);}   //Trigger a memcopy.
            for(int j = 0; j < N_d; j++) Qs[j].wait();
            times_memcpy[i-N_warmup] =  duration<double,nano>(steady_clock::now() - start).count();
            for(int j = 0; j < N_d; j++) {dualise_sycl_v0<6>(Qs[j], batches[j], LaunchPolicy::SYNC);}   //Dualise the batch.
            for(int j = 0; j < N_d; j++) Qs[j].wait();
            times_dual[i-N_warmup] =  duration<double,nano>(steady_clock::now() - start).count();
            for(int j = 0; j < N_d; j++) {tutte_layout_sycl(Qs[j], batches[j], LaunchPolicy::SYNC);}   //Compute the Tutte layout.
            for(int j = 0; j < N_d; j++) Qs[j].wait();
            times_tutte[i-N_warmup] =  duration<double,nano>(steady_clock::now() - start).count();
            for(int j = 0; j < N_d; j++) {spherical_projection_sycl(Qs[j], batches[j], LaunchPolicy::SYNC);}   //Project the layout onto a sphere.
            for(int j = 0; j < N_d; j++) Qs[j].wait();
            times_project[i-N_warmup] =  duration<double,nano>(steady_clock::now() - start).count();
            for(int j = 0; j < N_d; j++) {forcefield_optimise_sycl(Qs[j], batches[j], N*5, N*5, LaunchPolicy::SYNC);}   //Forcefield optimise the layout.
            for(int j = 0; j < N_d; j++) Qs[j].wait();
            times_opt[i-N_warmup] =  duration<double,nano>(steady_clock::now() - start).count();
        } else {
            for(int j = 0; j < N_d; j++) fill(batches[j]);
            for(int j = 0; j < N_d; j++) {nop_kernel(Qs[j], batches[j], LaunchPolicy::SYNC);}   //Dualise the batch.
            for(int j = 0; j < N_d; j++) {dualise_sycl_v0<6>(Qs[j], batches[j], LaunchPolicy::SYNC);}   //Dualise the batch.
            for(int j = 0; j < N_d; j++) {tutte_layout_sycl(Qs[j], batches[j], LaunchPolicy::SYNC);}   //Dualise the batch.
            for(int j = 0; j < N_d; j++) {spherical_projection_sycl(Qs[j], batches[j], LaunchPolicy::SYNC);}   //Dualise the batch.
            for(int j = 0; j < N_d; j++) {forcefield_optimise_sycl(Qs[j], batches[j], N*5, N*5, LaunchPolicy::SYNC);}   //Dualise the batch.
            for(int j = 0; j < N_d; j++) Qs[j].wait();
        }
    }

    /* file
        << N << ","
        << N_graphs << ","
        << mean(times_fill)/N_graphs << ","
        << stddev(times_fill)/N_graphs << ","
        << mean(times_memcpy)/N_graphs << ","
        << stddev(times_memcpy)/N_graphs << ","
        << mean(times_dual)/N_graphs << ","
        << stddev(times_dual)/N_graphs << ","
        << mean(times_tutte)/N_graphs << ","
        << stddev(times_tutte)/N_graphs << ","
        << mean(times_project)/N_graphs << ","
        << stddev(times_project)/N_graphs << ","
        << mean(times_opt)/N_graphs << ","
        << stddev(times_opt)/N_graphs << "\n"; */
    cout << "Fill: " << mean(times_fill)/N_graphs << " +- " << stddev(times_fill)/N_graphs << endl;
    cout << "Memcpy: " << mean(times_memcpy)/N_graphs << " +- " << stddev(times_memcpy)/N_graphs << endl;
    cout << "Dual: " << mean(times_dual)/N_graphs << " +- " << stddev(times_dual)/N_graphs << endl;
    cout << "Tutte: " << mean(times_tutte)/N_graphs << " +- " << stddev(times_tutte)/N_graphs << endl;
    cout << "Project: " << mean(times_project)/N_graphs << " +- " << stddev(times_project)/N_graphs << endl;
    cout << "Opt: " << mean(times_opt)/N_graphs << " +- " << stddev(times_opt)/N_graphs << endl;
    return 0;
}