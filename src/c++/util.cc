#include <numeric>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <type_traits>
#include <limits>
#include <iostream>
#define CPPBATCH
#include "util.h"
#include "cpp_kernels.h"

template float mean(std::vector<float> const& v);
template double mean(std::vector<double> const& v);

template float stddev(std::vector<float> const& data);
template double stddev(std::vector<double> const& data);

template void remove_outliers(std::vector<float>& data, int n_sigma);
template void remove_outliers(std::vector<double>& data, int n_sigma);

template void fill(IsomerBatch<float,uint16_t>& B, int set_div, int offset);

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

template <typename T, typename K>
void fill(IsomerBatch<T,K>& B, int set_div, int offset) {
  auto N = B.n_atoms;
  auto Nf = B.n_faces;
  auto N_graphs = B.m_capacity;

  const std::string path = cwd() + "/isomerspace_samples/dual_layout_" + std::to_string(N) + "_seed_42";
  std::ifstream samples(path, std::ios::binary);        //Open the file containing the samples.
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
