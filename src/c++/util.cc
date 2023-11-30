#include <numeric>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <type_traits>
#include <limits>
#include <iostream>
#define CPPBATCH
#include "config.hh"
#include "util.h"
#include "cpp_kernels.h"


//template float mean(std::vector<float> const& v);
//template double mean(std::vector<double> const& v);

template float stddev(std::vector<float> const& data);
template double stddev(std::vector<double> const& data);

template void remove_outliers(std::vector<float>& data, int n_sigma);
template void remove_outliers(std::vector<double>& data, int n_sigma);

template void fill(IsomerBatch<float,uint16_t>& B, int set_div, int offset);
template void fill(IsomerBatch<double,uint16_t>& B, int set_div, int offset);

template void bucky_fill(IsomerBatch<float,uint16_t>& B, int ntasks, int mytask_id);
template void bucky_fill(IsomerBatch<double,uint16_t>& B, int ntasks, int mytask_id);

/* template <typename T>
T mean(const std::vector<T>& v) {
  T sum = 0;
  for(size_t i=0;i<v.size();i++) sum += v[i];
  return sum/v.size();
}
 */
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

template <typename T, typename K>
void fill(IsomerBatch<T,K>& B, int set_div, int offset) {
  auto N = B.n_atoms;
  auto Nf = B.n_faces;
  auto N_graphs = B.m_capacity;

  const std::string path = std::string(SAMPLES_PATH) + "/dual_layout_" + std::to_string(N) + "_seed_42";
  std::ifstream samples(path, std::ios::binary);        //Open the file containing the samples.
  if(samples.fail())
    throw std::runtime_error("Could not open "+path+" for reading.\n");
  
  size_t fsize = filesize(samples);                     //Get the size of the file in bytes.
  size_t n_samples = fsize / (Nf * 6 * sizeof(K));   //All the graphs are fullerene graphs stored in 16bit unsigned integers.

  std::vector<K> in_buffer(n_samples * Nf * 6);   //Allocate a buffer to store all the samples.
  samples.read((char*)in_buffer.data(), n_samples*Nf*6*sizeof(K));         //Read all the samples into the buffer.

  for(int i = 0; i < N_graphs; i++) {                  //Copy the first N_graphs samples into the batch.
    B.statuses[i] = IsomerStatus::NOT_CONVERGED;
    for(int j = 0; j < Nf; j++) {
      for(int k = 0; k < 6; k++) {
	      (B.dual_neighbours).at(i*Nf*6 + j*6 + k) = in_buffer[(i%n_samples)*Nf*6 + j*6 + k];
	      if(k==5) (B.face_degrees).at(i*Nf + j) = in_buffer[(i%n_samples)*Nf*6 + j*6 + k] == std::numeric_limits<K>::max() ? 5 : 6;
      }
    }
  }
}

template <typename T, typename K>
void bucky_fill(IsomerBatch<T,K>& B, int ntasks, int mytask_id) {
  int N = B.n_atoms;
  int Nf = B.n_faces;
  int N_graphs = B.m_capacity;

  BuckyGen::buckygen_queue BuckyQ = BuckyGen::start(N,false,false, mytask_id, ntasks);  
  Graph G;

  G.neighbours = neighbours_t(Nf, std::vector<node_t>(6));
  G.N = Nf;
  int num_generated = 0;
  for (int i = 0; i < N_graphs; ++i) {
    bool more_isomers = BuckyGen::next_fullerene(BuckyQ, G);
    if (!more_isomers) break;
    num_generated++;
    B.statuses[i] = IsomerStatus::NOT_CONVERGED;
    for(int j = 0; j < Nf; j++) {
      B.face_degrees[i*Nf + j] = G.neighbours[j].size();
      for(int k = 0; k < G.neighbours[j].size(); k++) {
        B.dual_neighbours[i*Nf*6 + j*6 + k] = G.neighbours[j][k];
      }
    }
  }
  if (num_generated < N_graphs) {
    for (int i = num_generated; i < N_graphs; ++i) {
      B.statuses[i] = IsomerStatus::NOT_CONVERGED;
      //Repeat the same graphs as already generated.
      for(int j = 0; j < Nf; j++) {
        B.face_degrees[i*Nf + j] = B.face_degrees[(i%num_generated)*Nf + j];
        for(int k = 0; k < 6; k++) {
          B.dual_neighbours[i*Nf*6 + j*6 + k] = B.dual_neighbours[(i%num_generated)*Nf*6 + j*6 + k];
        }
      }
    }
  }

}


