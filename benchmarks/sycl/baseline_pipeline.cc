#include "fstream"
#include "iostream"
#include "vector"
#include "chrono"
#include "numeric"
#include "cmath"
#include "algorithm"
#define SYCLBATCH
#define IFW(x) if(x >= N_warmup)
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
    int N_gpus = argc > 7 ? stoi(argv[7]) : 1;
    string filename = argc > 8 ? argv[8] : "multi_"+device_type+".csv";
    
    cout << "Dualising " << N_graphs << " triangulation graphs, each with " << N
	 << " triangles, repeated " << N_runs << " times and with " << N_warmup
	 << " warmup runs." << endl;

    // Get all appropriate devices
    int N_d = 0;
    int Nf = N/2 + 2;
    vector<sycl::queue> Qs;
    for (auto platform : sycl::platform::get_platforms())
    {
        cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << endl;

        for (auto device : platform.get_devices())
        {
	  
            if (device.is_gpu()){
                N_d++;
                Qs.push_back(sycl::queue(device));
            }
            printf("\t%s: %s has %d compute-units.\n",
                device.is_gpu()? "USING    ":"NOT USING",
                device.get_info<sycl::info::device::name>().c_str(),
                device.get_info<sycl::info::device::max_compute_units>());
        }
    }
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
    if(file_check.peek() == ifstream::traits_type::eof()) file << "N,BS,T_gen,TSD_gen,T_overhead,TSD_overhead,T_dual,TSD_dual,T_tutte,TSD_tutte,T_project,TSD_project,T_opt,TSD_opt\n";
    auto selector =  device_type == "cpu" ? sycl::cpu_selector_v : sycl::gpu_selector_v;

    sycl::queue Q = sycl::queue(selector, sycl::property::queue::in_order{});
    vector<IsomerBatch<float,uint16_t>> batches; for(int i = 0; i < N_d; i++) batches.push_back(IsomerBatch<float,uint16_t>(N, N_graphs/N_d + N_graphs%N_d));
    Graph G(N);
    auto fill_and_dualise = [&](IsomerBatch<float,uint16_t>& batch) -> std::tuple<double, double, double>
    {
    double overhead = 0;
    double buckytime = 0; double dualtime = 0;
    auto start = std::chrono::steady_clock::now();
    BuckyGen::buckygen_queue BuckyQ = BuckyGen::start(N, 0, 0);

    sycl::host_accessor acc_dual(batch.dual_neighbours, sycl::write_only);
    sycl::host_accessor acc_cubic(batch.cubic_neighbours, sycl::write_only);
    sycl::host_accessor acc_degs(batch.face_degrees, sycl::write_only);
    sycl::host_accessor acc_status (batch.statuses, sycl::write_only);

    overhead += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    for (size_t ii = 0; ii < batch.m_capacity; ii++)
    {   
        auto start = std::chrono::steady_clock::now();
        auto more = BuckyGen::next_fullerene(BuckyQ, G);
        if (ii == 0){ overhead += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count(); }
        else{ buckytime += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count(); }
        auto T0 = std::chrono::steady_clock::now();
        for (size_t j = 0; j < Nf; j++)
        {
            for(size_t k = 0; k < G.neighbours[j].size(); k++)
            {
                acc_dual[ii*Nf*6 + j*6 + k] = G.neighbours[j][k];
            } 
            if(G.neighbours[j].size() == 5){
                acc_dual[ii*Nf*6 + j*6 + 5] = std::numeric_limits<uint16_t>::max();
                acc_degs[ii*Nf + j] = 5;
            } else {
                acc_degs[ii*Nf + j] = 6;
            }   

        }
        overhead += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - T0).count();
        buckytime += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        FullereneDual FD(G);
        auto T1 = std::chrono::steady_clock::now();
        FD.update();
        PlanarGraph pG = FD.dual_graph();
        auto T2 = std::chrono::steady_clock::now(); dualtime += std::chrono::duration<double, std::nano>(T2 - T1).count();

        for (size_t j = 0; j < N; j++){
            for (size_t k = 0; k < 3; k++)
            {
                acc_cubic[ii*N*3 + j*3 + k] = pG.neighbours[j][k];
            }
            
        }
        if(!more) break;

        
        acc_status[ii] = IsomerStatus::NOT_CONVERGED;
        auto T3 = std::chrono::steady_clock::now(); overhead += std::chrono::duration<double, std::nano>(T3 - T2).count();
    }
        BuckyGen::stop(BuckyQ);
        return {overhead, buckytime, dualtime};
    };
    
    vector<double> times_generate(N_runs); //Times in nanoseconds.
    vector<double> times_overhead(N_runs); //Times in nanoseconds.
    vector<double> times_dual(N_runs); //Times in nanoseconds.
    vector<double> times_tutte(N_runs); //Times in nanoseconds.
    vector<double> times_project(N_runs); //Times in nanoseconds.
    vector<double> times_opt(N_runs); //Times in nanoseconds.

    for(int i = 0; i < N_runs + N_warmup; i++){
        IFW(i) { 
            for(int j = 0; j < N_gpus; j++) {
                auto times = fill_and_dualise(batches[j]);
                times_overhead[i - N_warmup] = std::get<0>(times);
                times_generate[i - N_warmup] = std::get<1>(times);
                times_dual[i - N_warmup] = std::get<2>(times);
            } 
        } else { 
            for(int j = 0; j < N_gpus; j++) fill_and_dualise(batches[j]); 
        }
        auto T0 = std::chrono::steady_clock::now();
        for(int j = 0; j < N_gpus; j++) nop_kernel(Qs[j], batches[j], LaunchPolicy::ASYNC);
        for(int j = 0; j < N_gpus; j++) {Qs[j].wait();} 
        auto T1 = std::chrono::steady_clock::now(); IFW(i) times_overhead[i - N_warmup] = std::chrono::duration<double, std::nano>(T1 - T0).count();
        for(int j = 0; j < N_gpus; j++) tutte_layout_sycl(Qs[j], batches[j], LaunchPolicy::ASYNC);
        for(int j = 0; j < N_gpus; j++) {Qs[j].wait();} 
        auto T2 = std::chrono::steady_clock::now(); IFW(i) times_tutte[i - N_warmup] = std::chrono::duration<double, std::nano>(T2 - T1).count();
        for(int j = 0; j < N_gpus; j++) spherical_projection_sycl(Qs[j], batches[j], LaunchPolicy::ASYNC);
        for(int j = 0; j < N_gpus; j++) {Qs[j].wait();} 
        auto T3 = std::chrono::steady_clock::now(); IFW(i) times_project[i - N_warmup] = std::chrono::duration<double, std::nano>(T3 - T2).count();
        for(int j = 0; j < N_gpus; j++) forcefield_optimise_sycl(Qs[j], batches[j], 5*N, 5*N, LaunchPolicy::ASYNC);
        for(int j = 0; j < N_gpus; j++) {Qs[j].wait();} 
        auto T4 = std::chrono::steady_clock::now(); IFW(i) times_opt[i - N_warmup] = std::chrono::duration<double, std::nano>(T4 - T3).count();
    }

    remove_outliers(times_generate); //Removes data points that are more than 2 standard deviations from the mean. If there are less than 3 data points, this does nothing.
    remove_outliers(times_overhead); //Removes data points that are more than 2 standard deviations from the mean. If there are less than 3 data points, this does nothing.
    remove_outliers(times_dual); //Removes data points that are more than 2 standard deviations from the mean. If there are less than 3 data points, this does nothing.
    remove_outliers(times_tutte); //Removes data points that are more than 2 standard deviations from the mean. If there are less than 3 data points, this does nothing.
    remove_outliers(times_project); //Removes data points that are more than 2 standard deviations from the mean. If there are less than 3 data points, this does nothing.
    remove_outliers(times_opt); //Removes data points that are more than 2 standard deviations from the mean. If there are less than 3 data points, this does nothing.

    std::cout << "N, Nf, BatchSize, device_type, generate, memcpy, dual, tutte, project, opt" << std::endl;
    std::cout << "N: " << N << ", Nf: " << Nf << ", BatchSize: " << N_graphs << ", device_type: " << device_type << "\n";
    std::cout << "Generate: " << mean(times_generate)/(N_graphs-1) << ", " << stddev(times_generate)/(N_graphs-1) << " ns \n";
    std::cout << "Overhead: " << mean(times_overhead)/N_graphs << ", " << stddev(times_overhead)/N_graphs << " ns \n";
    std::cout << "Dual: " << mean(times_dual)/N_graphs << ", " << stddev(times_dual)/N_graphs << " ns \n";
    std::cout << "Tutte: " << mean(times_tutte)/N_graphs << ", " << stddev(times_tutte)/N_graphs << " ns \n";
    std::cout << "Project: " << mean(times_project)/N_graphs << ", " << stddev(times_project)/N_graphs << " ns \n";
    std::cout << "Opt: " << mean(times_opt)/N_graphs << ", " << stddev(times_opt)/N_graphs << " ns \n";
    
    file << N << ","
         << N_graphs << ","
         << mean(times_generate)/N_graphs << ","
         << stddev(times_generate)/N_graphs << ","
         << mean(times_overhead)/N_graphs << ","
         << stddev(times_overhead)/N_graphs << ","
         << mean(times_dual)/N_graphs << ","
         << stddev(times_dual)/N_graphs << ","
         << mean(times_tutte)/N_graphs << ","
         << stddev(times_tutte)/N_graphs << ","
         << mean(times_project)/N_graphs << ","
         << stddev(times_project)/N_graphs << ","
         << mean(times_opt)/N_graphs << ","
         << stddev(times_opt)/N_graphs << "\n";
    return 0;
}
