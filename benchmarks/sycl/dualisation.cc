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


vector<sycl::queue> get_device_queues(bool want_gpus){
  int N_d = 0;
  vector<sycl::queue> Qs;
  for (auto platform : sycl::platform::get_platforms()) 
    for (auto device : platform.get_devices()) {
      bool use_this_device = (want_gpus && device.is_gpu())  || (!want_gpus && device.is_cpu() &&
								 (N_d<1/* NB: Sometimes SYCL reports the same CPU twice, leading to death and pain. */));
      if (use_this_device){
	N_d++;
	Qs.push_back(sycl::queue(device));
	printf("Found %s device #%ld: %s with %d compute-units.\n",
	       platform.get_info<sycl::info::platform::name>().c_str(),
	       Qs.size(),
	       device.get_info<sycl::info::device::name>().c_str(),
	       device.get_info<sycl::info::device::max_compute_units>());
      }
    }
  return Qs;
}


int main(int argc, char** argv) {
    // Make user be explicit about which device we want to test, to avoid surprises.
    if(argc < 2 || (string(argv[1]) != "cpu" && string(argv[1]) != "gpu")){
      fprintf(stderr, "Syntax: %s <cpu|gpu> [N:200] [N_graphs:1000000] [N_runs:10] [N_warmup:1] [version:0] [filename:multi_gpu.csv]\n",argv[0]);
      return -1;
    }
    if(argc >6 && (stoi(argv[6]) < 0 || stoi(argv[6]) > 4)){
      fprintf(stderr, "Invalid kernel version %d, must be between 1 and 4.\n", stoi(argv[6]));
      return -1;
    }
    string device_type = argv[1];
    bool want_gpus = (device_type == "gpu");
      
    // Parameters for the benchmark run
    int N = argc > 2 ? stoi(argv[2]) : 200;
    int N_graphs = argc > 3 ? stoi(argv[3]) : 1000000;
    int N_runs = argc > 4 ? stoi(argv[4]) : 10;
    int N_warmup = argc > 5 ? stoi(argv[5]) : 1;
    int version = argc > 6 ? stoi(argv[6]) : 1;
    int N_gpus = argc > 7 ? stoi(argv[7]) : 1;
    string filename = argc > 8 ? argv[8] : "multi_"+device_type+".csv";
    
    cout << "Dualising " << N_graphs << " triangulation graphs, each with " << N
	 << " triangles, repeated " << N_runs << " times and with " << N_warmup
	 << " warmup runs." << endl;

    // Get all appropriate devices

    vector<sycl::queue> Qs = get_device_queues(want_gpus);
    int N_d = Qs.size();
    
    if (N_d < N_gpus)
    {
        cout << "Not enough devices available. Exiting." << endl;
        return 1;
    }
    printf("Using %d %s-devices\n",N_gpus,device_type.c_str());

    if(N==22 || N%2==1 || N<20 || N>200){
        cout << "N must be even and between 20 and 200 and not equal to 22." << endl;
        return 1;
    }

    ifstream file_check(filename);
    ofstream file(filename, ios_base::app);
    //If the file is empty, write the header.
    if(file_check.peek() == ifstream::traits_type::eof()) file << "N,BS,T,TSD\n";

    int Nf = N/2 + 2;
    N_d = N_gpus;

    //vector<IsomerBatch<float,uint16_t>> batches= {IsomerBatch<float,uint16_t>(N, N_graphs/N_d + N_graphs%N_d), IsomerBatch<float,uint16_t>(N, N_graphs/N_d + N_graphs%N_d)};

    vector<IsomerBatch<float,uint16_t>> batches; for(int i = 0; i < N_d; i++) batches.push_back(IsomerBatch<float,uint16_t>(N, N_graphs/N_d + N_graphs%N_d));

    
    for(int i = 0; i < N_d; i++) bucky_fill(batches[i],i,N_d);

    vector<double> times(N_runs); //Times in nanoseconds.
    for (size_t i = 0; i < (N_runs + N_warmup); i++)
    {
        if(i >= N_warmup) {
            auto start = steady_clock::now();
            switch (version)
            {
            case 1:
                for(int j = 0; j < N_d; j++) {dualise_sycl_v1<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);}   //Dualise the batch.
                break;
            case 2:
                for(int j = 0; j < N_d; j++) {dualise_sycl_v2<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);}    //Dualise the batch.
                break;
            case 3:
                for(int j = 0; j < N_d; j++) {dualise_sycl_v3<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);}    //Dualise the batch.
                break;
            case 4:
                for(int j = 0; j < N_d; j++) {dualise_sycl_v4<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);}    //Dualise the batch.
                break;
            default:
                break;
            }
            for(int j = 0; j < N_d; j++) Qs[j].wait();
            times[i-N_warmup] = duration<double,nano>(steady_clock::now() - start).count();
        } else {
            switch (version)
            {
            case 1:
                for(int j = 0; j < N_d; j++) dualise_sycl_v1<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);    //Dualise the batch.
                break;
            case 2:
                for(int j = 0; j < N_d; j++) dualise_sycl_v2<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);    //Dualise the batch.
                break;
            case 3:
                for(int j = 0; j < N_d; j++) dualise_sycl_v3<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);    //Dualise the batch.
                break;
            case 4:
                for(int j = 0; j < N_d; j++) dualise_sycl_v4<6>(Qs[j], batches[j], LaunchPolicy::ASYNC);    //Dualise the batch.
                break;
            default:
                break;
            }
            for (int j = 0; j < N_d; j++) Qs[j].wait();
        }
    }

    remove_outliers(times); //Removes data points that are more than 2 standard deviations from the mean. If there are less than 3 data points, this does nothing.

    file
        << N << ","
        << N_graphs << ","
        << mean(times)/N_graphs << ","
        << stddev(times)/N_graphs << "\n";
    cout << "  Mean Time per Graph: " << mean(times) / N_graphs << " +/- " << stddev(times) / N_graphs << " ns\n";
    return 0;
}
