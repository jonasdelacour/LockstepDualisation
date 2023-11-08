#define CUDA_ENABLED 1

#include "include/dual.h"
#include "fstream"
#include "iostream"
#include "cu_array.h"
#include "launch_ctx.h"
#include "vector"
#include "chrono"
#include "numeric"
#include "cmath"
#include "algorithm"
#include "mpi.h"
#include "unistd.h"
#include "limits.h"
using namespace std::chrono_literals;
using namespace std::chrono;

#include <numeric>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <unistd.h>

#include "util.h"
#include "dual.h"

template float mean(std::vector<float> const& v);
template double mean(std::vector<double> const& v);

template float stddev(std::vector<float> const& data);
template double stddev(std::vector<double> const& data);

template void remove_outliers(std::vector<float>& data, int n_sigma);
template void remove_outliers(std::vector<double>& data, int n_sigma);



template <typename T>
T mean(const std::vector<T>& v) {
  T sum = 0;
  for(size_t i=0;i<v.size();i++) sum += v[i];
  return sum/v.size();
}

template<typename T>
T stddev(const std::vector<T>& data)
{
  if(data.size() <= 1) return 0;
  
    // Calculate the mean
    T mn = mean(data);

    // Calculate the sum of squared differences from the mean
    T sum_of_squares = 0.0;
    for (const T& value : data)
    {
        T diff = value - mn;
        sum_of_squares += diff * diff;
    }

    // Calculate the variance and return the square root
    T variance = sum_of_squares / (data.size() - 1);
    return std::sqrt(variance);
}

template<typename T>
void remove_outliers(std::vector<T>& data, int n_sigma) {
    if (data.size() < 3) return;
    std::sort(data.begin(), data.end());
    T mean_ = mean(data);
    T stddev_ = stddev(data);
    T lower_bound = mean_ - n_sigma*stddev_;
    T upper_bound = mean_ + n_sigma*stddev_;
    data.erase(std::remove_if(data.begin(), data.end(), [lower_bound, upper_bound](T x) { return x < lower_bound || x > upper_bound; }), data.end());
}

size_t filesize(std::ifstream &f)
{
  f.seekg(0,f.end);
  size_t n = f.tellg();
  f.seekg(0,f.beg);

  return n;
}

std::string cwd()
{
  char p[0x1000];
  if(!getcwd(p,0x1000)){
    perror("getcwd()");
    abort();
  }
  return std::string(p);
}

template <typename T, typename U>
void fill(T& G_in, U& degrees, const int Nf, const int N_graphs, int set_div,int offset) {
  int N = (Nf - 2)*2;

  const std::string path = cwd() + "/isomerspace_samples/dual_layout_" + std::to_string(N) + "_seed_42";
  std::ifstream samples(path, std::ios::binary);         //Open the file containing the samples.
  int fsize = filesize(samples);                          //Get the size of the file in bytes.
  int n_samples = fsize / (Nf * 6 * sizeof(uint16_t));   //All the graphs are fullerene graphs stored in 16bit unsigned integers.

  std::vector<uint16_t> in_buffer(n_samples * Nf * 6);   //Allocate a buffer to store all the samples.
  samples.read((char*)in_buffer.data(), n_samples*Nf*6*sizeof(uint16_t));         //Read all the samples into the buffer.

  for(int i = 0; i < N_graphs; i++) {                  //Copy the first N_graphs samples into the batch.
    for(int j = 0; j < Nf; j++) {
      for(int k = 0; k < 6; k++) {
	G_in[i*Nf*6 + j*6 + k] = in_buffer[(i%(n_samples/set_div) + offset)*Nf*6 + j*6 + k];
	if(k==5) degrees[i*Nf + j] = in_buffer[(i%(n_samples/set_div) + offset)*Nf*6 + j*6 + k] == UINT16_MAX ? 5 : 6;
      }
    }
  }
}

template void fill(std::vector<d_node_t>& G_in, std::vector<uint8_t>& degrees, const int Nf, const int N_graphs, int set_div=1, int offset=0);


void execute_kernel(const std::vector<CuArray<uint16_t>>& in_graphs,const std::vector<CuArray<uint8_t>>& in_degrees, std::vector<CuArray<uint16_t>>& out_graphs, const int version, const int Nf, const int N_graphs){
    int N_d = in_graphs.size();
    std::vector<LaunchCtx> ctxs(N_d); for(int i = 0; i < N_d; i++) {ctxs[i] = LaunchCtx(i);}
    switch (version)
    {
    case 0:
        for(int j = 0; j < N_d; j++) {dualise_V0<6>(in_graphs[j], in_degrees[j], out_graphs[j], Nf, N_graphs/N_d, ctxs[j], LaunchPolicy::ASYNC);}   //Dualise the batch.
        break;
    case 1:
        for(int j = 0; j < N_d; j++) {dualise_V1<6>(in_graphs[j], in_degrees[j], out_graphs[j], Nf, N_graphs/N_d, ctxs[j], LaunchPolicy::ASYNC);}    //Dualise the batch.
        break;
    default:
        break;
    }
    for(int j = 0; j < N_d; j++) {ctxs[j].wait();}
}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = argc > 1 ? std::stoi(argv[1]) : 200;
    int N_graphs = argc > 2 ? std::stoi(argv[2]) : 1000000;
    int N_runs = argc > 3 ? std::stoi(argv[3]) : 10;
    int N_warmup = argc > 4 ? std::stoi(argv[4]) : 1;
    int version = argc > 5 ? std::stoi(argv[5]) : 0;
    std::string filename = argc > 6 ? argv[6] : "multi_node_multi_gpu.csv";
    
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0) {
        std::cout << "Dualising " << N_graphs << " triangulation graphs, each with " << N
                  << " triangles, repeated " << N_runs << " times and with " << N_warmup
                  << " warmup runs." << std::endl;
    }
    
    int N_d = LaunchCtx::get_device_count();
    if((N==22 || N%2==1 || N<20 || N>200) && world_rank == 0){
        std::cout << "N must be even and between 20 and 200 and not equal to 22." << std::endl;
        return 1;
    }

    std::vector<LaunchCtx> ctxs(N_d);
    

    std::ifstream file_check(filename);
    std::ofstream file(filename, std::ios_base::app);
    //If the file is empty, write the header.
    if(file_check.peek() == std::ifstream::traits_type::eof()) file << "N,BS,T,TSD,TD,TDSD\n";

    int Nf = N/2 + 2;
    char pname[HOST_NAME_MAX];
    gethostname(pname, HOST_NAME_MAX);
    std::vector<CuArray<uint16_t>> in_graphs(N_d); for(int i = 0; i < N_d; i++) {in_graphs[i]=CuArray<uint16_t>((N_graphs/N_d + N_graphs%N_d)*Nf*6);}
    std::vector<CuArray<uint8_t>> in_degrees(N_d); for(int i = 0; i < N_d; i++) {in_degrees[i]=CuArray<uint8_t>((N_graphs/N_d + N_graphs%N_d)*Nf);}
    std::vector<CuArray<uint16_t>> out_graphs(N_d); for(int i = 0; i < N_d; i++) {out_graphs[i]=CuArray<uint16_t>((N_graphs/N_d + N_graphs%N_d)*N*3);}
    auto send_recv = [&]{
      if (world_rank == 0) {
        for(int i = 0; i < N_d; i++) fill(in_graphs[i], in_degrees[i], Nf, N_graphs/N_d + N_graphs%N_d);
        for (int i = 0; i < N_d; i++) {
            MPI_Send(in_graphs[i].data, (N_graphs/N_d + N_graphs%N_d)*Nf*6, MPI_SHORT, 1, 0, MPI_COMM_WORLD);
            MPI_Send(in_degrees[i].data, (N_graphs/N_d + N_graphs%N_d)*Nf, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        }
      } else {
          for (int i = 0; i < N_d; i++) {
              MPI_Recv(in_graphs[i].data, (N_graphs/N_d + N_graphs%N_d)*Nf*6, MPI_SHORT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              MPI_Recv(in_degrees[i].data, (N_graphs/N_d + N_graphs%N_d)*Nf, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
      }
    };
    
    auto fillin = [&]{
      if (world_rank == 0) {
        for(int i = 0; i < N_d; i++) fill(in_graphs[i], in_degrees[i], Nf, N_graphs/N_d + N_graphs%N_d, 2, 0);
      }
      else {
        for(int i = 0; i < N_d; i++) fill(in_graphs[i], in_degrees[i], Nf, N_graphs/N_d + N_graphs%N_d, 2, 5000);
      }
    };
    


    fillin();
    for(int i = 0; i < N_d; i++) {
        in_graphs[i].to_device(i);
        in_degrees[i].to_device(i);
        out_graphs[i].to_device(i);
    }
    for (int I = 0; I < N_warmup; I++) {
        execute_kernel(in_graphs, in_degrees, out_graphs,version, Nf, N_graphs);
    }
    std::vector<double> times(N_runs); //Times in nanoseconds.
    std::vector<double> tdiffs(N_runs); //Timing differences in nanoseconds.
    for (int I = 0; I < N_runs; I++){
      auto start = std::chrono::high_resolution_clock::now();
      execute_kernel(in_graphs, in_degrees, out_graphs,version, Nf, N_graphs);
      auto end = std::chrono::high_resolution_clock::now();
      times[I] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

//execute_kernel(in_graphs, in_degrees, out_graphs,version, Nf, N_graphs);
    double result = 0.;
    double send_buf = mean(times);
    auto err = MPI_Reduce(&send_buf, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
      remove_outliers(times);
        std::cout << pname << ":  Mean Time per Graph: " << result / (N_graphs * world_size) << " +/- " << stddev(times) / N_graphs << " ns" << std::endl;
      file
        << N << ","
        << N_graphs << ","
        << mean(times)/N_graphs << ","
        << stddev(times)/N_graphs << ","
        << mean(tdiffs)/N_graphs << ","
        << stddev(tdiffs)/N_graphs << "\n";
    } else {
       // std::cout << pname <<":   Mean Time per Graph: " << mean(times) / N_graphs << " +/- " << stddev(times) / N_graphs << " ns" << std::endl;
    }


    // Initialize the MPI environment


    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}
