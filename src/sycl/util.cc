#include <fstream>
#include <iostream>
#include <unistd.h>
#define SYCLBATCH
#include "config.hh"
#include "util.h"
#include "sycl_kernels.h"
using namespace sycl;



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
void fill(IsomerBatch<T,K>& B, int set_div,int offset) {
    int N = B.n_atoms;
    int Nf = B.n_faces;
    int N_graphs = B.m_capacity;
    auto face_degrees_acc = host_accessor(B.face_degrees);
    auto dual_neighbours_acc = host_accessor(B.dual_neighbours);
    auto statuses_acc = host_accessor(B.statuses);


    const std::string path = std::string(SAMPLES_PATH) + "/dual_layout_" + std::to_string(N) + "_seed_42";
    std::ifstream samples(path, std::ios::binary);         //Open the file containing the samples.
    if(samples.fail())
      throw std::runtime_error("Could not open "+path+" for reading.\n");
    
    int fsize = filesize(samples);                          //Get the size of the file in bytes.
    int n_samples = fsize / (Nf * 6 * sizeof(uint16_t));   //All the graphs are fullerene graphs stored in 16bit unsigned integers.

    std::vector<uint16_t> in_buffer(n_samples * Nf * 6);   //Allocate a buffer to store all the samples.
    samples.read((char*)in_buffer.data(), n_samples*Nf*6*sizeof(uint16_t));         //Read all the samples into the buffer.

    for(int i = 0; i < N_graphs; i++) {                  //Copy the first N_graphs samples into the batch.
      statuses_acc[i] = IsomerStatus::NOT_CONVERGED;
      for(int j = 0; j < Nf; j++) {
        for(int k = 0; k < 6; k++) {
    	dual_neighbours_acc[i*Nf*6 + j*6 + k] = in_buffer[(i%(n_samples/set_div) + offset)*Nf*6 + j*6 + k];
    	if(k==5) face_degrees_acc[i*Nf + j] = in_buffer[(i%(n_samples/set_div) + offset)*Nf*6 + j*6 + k] == UINT16_MAX ? 5 : 6;
        }
      }
    }
}
template void fill(IsomerBatch<float,uint16_t>& B, int set_div, int offset);
