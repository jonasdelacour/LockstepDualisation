#include "fstream"
#include "iostream"
#include "vector"
#include "chrono"
#include "numeric"
#include "cmath"
#include "algorithm"
#include "random"
#define SYCLBATCH
#define IFW(x) if(x >= N_warmup)
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
	printf("Using %s device %s with %d compute-units.\n",
	       platform.get_info<sycl::info::platform::name>().c_str(),		   
	       device.get_info<sycl::info::device::name>().c_str(),
	       device.get_info<sycl::info::device::max_compute_units>());
      }
    }
  return Qs;
}


int main(int argc, char** argv) {
    // Make user be explicit about which device we want to test, to avoid surprises.
    if(argc < 2 || (string(argv[1]) != "cpu" && string(argv[1]) != "gpu")){
      fprintf(stderr, "Syntax: %s <cpu|gpu> [N:200] [batch_size:10000] [N_runs:10] [N_warmup:1] [version:0] [N_devices:1] [filename:multi_gpu.csv]\n",argv[0]);
      return -1;
    }
    string device_type = argv[1];
    bool want_gpus = (device_type == "gpu");
    
    // Parameters for the benchmark run
    int N = argc > 2 ? stoi(argv[2]) : 200;
    int batch_size = argc > 3 ? stoi(argv[3]) : 10000;
    int N_runs = argc > 4 ? stoi(argv[4]) : 10;
    int N_warmup = argc > 5 ? stoi(argv[5]) : 1;
    int version = argc > 6 ? stoi(argv[6]) : 0;
    int N_gpus = argc > 7 ? stoi(argv[7]) : 1;
    string filename = argc > 8 ? argv[8] : "multi_"+device_type+".csv";


    if(N==22 || N%2==1 || N<20 || N>200){
        cout << "N must be even and between 20 and 200 and not equal to 22." << endl;
        return 1;
    }    
    
    size_t M_graphs = min<size_t>((N_runs+N_warmup)*batch_size, (int)IsomerDB::number_isomers(N, "Any", false));
    batch_size      = min<size_t>(batch_size, M_graphs);    
    N_warmup        = (M_graphs>=2*batch_size);
    N_runs          = M_graphs/batch_size - N_warmup; 

    cout << "Dualizing " << batch_size << " triangulation graphs, each with " << N
	 << " triangles, repeated " << N_runs << " times and with " << N_warmup
	 << " warmup runs." << endl;

    BuckyGen::buckygen_queue BuckyQ = BuckyGen::start(N);
    
    // Get all appropriate devices
    int Nf = N/2 + 2;
    vector<sycl::queue> Qs = get_device_queues(want_gpus);
    int N_d = Qs.size();

    if (N_d < N_gpus)    {
        cout << "Not enough devices available. Exiting." << endl;
        return 1;
    } else
      printf("Using %d %s-devices\n",N_gpus,device_type.c_str());
 
    ifstream file_check(filename);
    ofstream file(filename, ios_base::app);
    //If the file is empty, write the header.
    if(file_check.peek() == ifstream::traits_type::eof()) file << "N,BS,T_gen,TSD_gen,T_overhead,TSD_overhead,T_dual,TSD_dual,T_tutte,TSD_tutte,T_project,TSD_project,T_opt,TSD_opt\n";
    auto selector =  device_type == "cpu" ? sycl::cpu_selector_v : sycl::gpu_selector_v;

    sycl::queue Q = sycl::queue(selector, sycl::property::queue::in_order{});
    vector<IsomerBatch<float,uint16_t>> batches; for(int i = 0; i < N_d; i++) batches.push_back(IsomerBatch<float,uint16_t>(N, batch_size/N_d + batch_size%N_d));
    typedef duration<double,nanoseconds::period> ns_time;
    
    auto fill_and_dualise = [&](IsomerBatch<float,uint16_t>& batch) -> std::tuple<double, double, double> {
      sycl::host_accessor acc_dual(batch.dual_neighbours, sycl::write_only);
      sycl::host_accessor acc_degs(batch.face_degrees, sycl::write_only);
      sycl::host_accessor acc_status (batch.statuses, sycl::write_only);
      sycl::host_accessor acc_cubic(batch.cubic_neighbours, sycl::write_only);
    
      Triangulation g;
      size_t isomer_ix = 0;
      steady_clock::time_point before_time, after_time;
      double overhead = 0;
      double buckytime = 0; double dualtime = 0;
      size_t ii = 0;
      while( BuckyGen::next_fullerene(BuckyQ,g) && (isomer_ix < M_graphs) && (ii<batch.m_capacity)){
	before_time = steady_clock::now();

	buckytime += ns_time(before_time-after_time).count(); // Time spent waiting for buckygen to generate g
	for (size_t j = 0; j < Nf; j++) {
	    for(size_t k = 0; k < g.neighbours[j].size(); k++)
		acc_dual[ii*Nf*6 + j*6 + k] = g.neighbours[j][k];

	    if(g.neighbours[j].size() == 5){
	      acc_dual[ii*Nf*6 + j*6 + 5] = std::numeric_limits<uint16_t>::max();
	      acc_degs[ii*Nf + j] = 5;
	    } else 
	      acc_degs[ii*Nf + j] = 6;
	}
	auto T0 = steady_clock::now();
	FullereneDual FD(g);
	auto T1 = steady_clock::now(); overhead += ns_time(T1 - T0).count();
	FD.update();
	PlanarGraph pG = FD.dual_graph();
	auto T2 = steady_clock::now(); dualtime += ns_time(T2 - T1).count();

	for (size_t j = 0; j < N; j++)
	  for (size_t k = 0; k < 3; k++)
	      acc_cubic[ii*N*3 + j*3 + k] = pG.neighbours[j][k];
                
	acc_status[ii] = IsomerStatus::NOT_CONVERGED;
	ii++;
	overhead += ns_time(steady_clock::now() - T2).count();

      isomer_ix++;
      after_time = steady_clock::now();
      }
      
      return {overhead, buckytime, dualtime};
    };


    vector<double> times_generate(N_runs,0); //Times in nanoseconds.
    vector<double> times_overhead(N_runs,0); //Times in nanoseconds.    
    vector<double> times_dual    (N_runs,0); //Times in nanoseconds.
    vector<double> times_tutte   (N_runs,0); //Times in nanoseconds.
    vector<double> times_project (N_runs,0); //Times in nanoseconds.
    vector<double> times_opt     (N_runs,0); //Times in nanoseconds.    


#define forcefield_optimise(Q,B,L) forcefield_optimise_sycl(Q, B, 5*N, 5*N, L)
#define sync_and_time(T) for(int j = 0; j < N_gpus; j++) Qs[j].wait(); auto T = steady_clock::now()
#define time_operation(kernel,name,t0,t1) \
    for(int j = 0; j < N_gpus; j++) kernel(Qs[j], batches[j], LaunchPolicy::ASYNC); \
    sync_and_time(t1);							            \
    IFW(i) name[i-N_warmup] += ns_time(t1-t0).count()
    
    for(int i = 0; i < N_runs + N_warmup; i++){
        auto T0 = steady_clock::now();
	time_operation(nop_kernel,  times_overhead,T0,T1);
	
        IFW(i) { 
            for(int j = 0; j < N_gpus; j++) {
	      auto [t_oh,t_gen,t_dual] = fill_and_dualise(batches[j]);
	      times_overhead[i - N_warmup] += t_oh;
	      times_generate[i - N_warmup] += t_gen;
	      times_dual    [i - N_warmup] += t_dual;
            } 
        } else 
            for(int j = 0; j < N_gpus; j++) fill_and_dualise(batches[j]); 
        auto T2 = steady_clock::now();
	time_operation(tutte_layout_sycl,        times_tutte,  T2,T3);
	time_operation(spherical_projection_sycl,times_project,T3,T4);
	time_operation(forcefield_optimise, times_opt,   T4,T5);	
    }

    BuckyGen::stop(BuckyQ);

    //Filter data points that are more than 2 standard deviations from the mean. If there are less than 3 data points, this does nothing.
    remove_outliers(times_generate);
    remove_outliers(times_overhead);
    remove_outliers(times_dual);    
    remove_outliers(times_tutte);   
    remove_outliers(times_project); 
    remove_outliers(times_opt);     

    std::cout << "N, Nf, BatchSize, device_type, generate, memcpy, dual, tutte, project, opt" << std::endl;
    std::cout << "N: " << N << ", Nf: " << Nf << ", BatchSize: " << batch_size << ", device_type: " << device_type << "\n";
    std::cout << "Generate: " << mean(times_generate)/(batch_size-1) << ", " << stddev(times_generate)/(batch_size-1) << " ns \n";
    std::cout << "Overhead: " << mean(times_overhead)/batch_size << ", " << stddev(times_overhead)/batch_size << " ns \n";
    std::cout << "Dual: " << mean(times_dual)/batch_size << ", " << stddev(times_dual)/batch_size << " ns \n";
    std::cout << "Tutte: " << mean(times_tutte)/batch_size << ", " << stddev(times_tutte)/batch_size << " ns \n";
    std::cout << "Project: " << mean(times_project)/batch_size << ", " << stddev(times_project)/batch_size << " ns \n";
    std::cout << "Opt: " << mean(times_opt)/batch_size << ", " << stddev(times_opt)/batch_size << " ns \n";
    
    file << N << ","
         << batch_size << ","
         << mean(times_generate)/batch_size << ","
         << stddev(times_generate)/batch_size << ","
         << mean(times_overhead)/batch_size << ","
         << stddev(times_overhead)/batch_size << ","
         << mean(times_dual)/batch_size << ","
         << stddev(times_dual)/batch_size << ","
         << mean(times_tutte)/batch_size << ","
         << stddev(times_tutte)/batch_size << ","
         << mean(times_project)/batch_size << ","
         << stddev(times_project)/batch_size << ","
         << mean(times_opt)/batch_size << ","
         << stddev(times_opt)/batch_size << "\n";
    return 0;
}
