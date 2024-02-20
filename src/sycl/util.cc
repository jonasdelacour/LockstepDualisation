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

// This will be very slow, as it will incur the (substantial) overhead of starting up BuckyGen for every batch.
// Also, it doesn't work with IPR or symmetric molecules.
template <typename T, typename K>
void bucky_fill(IsomerBatch<T,K>& B, int mytask_id, int ntasks) {
  int N = B.n_atoms;
  int Nf = B.n_faces;
  int N_graphs = B.m_capacity;
  auto face_degrees_acc = host_accessor(B.face_degrees);
  auto dual_neighbours_acc = host_accessor(B.dual_neighbours);
  auto statuses_acc = host_accessor(B.statuses);
  BuckyGen::buckygen_queue BuckyQ = BuckyGen::start(N,false, false, mytask_id, ntasks);
  Graph G;

  G.neighbours = neighbours_t(Nf, std::vector<node_t>(6,-1));
  G.N = Nf;
  int num_generated = 0;
  for (int i = 0; i < N_graphs; ++i) {
    bool more_isomers = BuckyGen::next_fullerene(BuckyQ, G);
    if (!more_isomers) break;
    num_generated++;
    statuses_acc[i] = IsomerStatus::NOT_CONVERGED;
    for(int j = 0; j < Nf; j++) {
      face_degrees_acc[i*Nf + j] = G.neighbours[j].size();
      for(int k = 0; k < G.neighbours[j].size(); k++) 
        dual_neighbours_acc[i*Nf*6 + j*6 + k] = G.neighbours[j][k];

    }
  }
  BuckyGen::stop(BuckyQ);
  if (num_generated < N_graphs) 
    for (int i = num_generated; i < N_graphs; ++i) {
      statuses_acc[i] = IsomerStatus::NOT_CONVERGED;
      //Repeat the same graphs as already generated.
      for(int j = 0; j < Nf; j++) {
        face_degrees_acc[i*Nf + j] = face_degrees_acc[(i%num_generated)*Nf + j];
        for(int k = 0; k < 6; k++) 
          dual_neighbours_acc[i*Nf*6 + j*6 + k] = dual_neighbours_acc[(i%num_generated)*Nf*6 + j*6 + k];

      }
    }
}

template <typename T, typename K>
bool bucky_fill(IsomerBatch<T,K>& B, BuckyGen::buckygen_queue& BuckyQ) {
  int N = B.n_atoms;
  int Nf = B.n_faces;
  int N_graphs = B.m_capacity;
  auto face_degrees_acc = host_accessor(B.face_degrees);
  auto dual_neighbours_acc = host_accessor(B.dual_neighbours);
  auto statuses_acc = host_accessor(B.statuses);
  Graph G;

  G.neighbours = neighbours_t(Nf, std::vector<node_t>(6,-1));
  G.N = Nf;
  int num_generated = 0;
  bool more_isomers = true;
  for (int i = 0; (i < N_graphs) && (more_isomers = BuckyGen::next_fullerene(BuckyQ, G)); ++i) {
    num_generated++;
    statuses_acc[i] = IsomerStatus::NOT_CONVERGED;
    for(int j = 0; j < Nf; j++) {
      const auto &nj = G.neighbours[j];
      face_degrees_acc[i*Nf + j] = nj.size();
      for(int k = 0; k < G.neighbours[j].size(); k++) 
        dual_neighbours_acc[i*Nf*6 + j*6 + k] = nj[k];

    }
  }
  
  if (num_generated < N_graphs) 
    for (int i = num_generated; i < N_graphs; ++i) {
      statuses_acc[i] = IsomerStatus::NOT_CONVERGED;
      //Repeat the same graphs as already generated.
      for(int j = 0; j < Nf; j++) {
        face_degrees_acc[i*Nf + j] = face_degrees_acc[(i%num_generated)*Nf + j];
        for(int k = 0; k < 6; k++) 
          dual_neighbours_acc[i*Nf*6 + j*6 + k] = dual_neighbours_acc[(i%num_generated)*Nf*6 + j*6 + k];

      }
    }

  return more_isomers;
}
template <typename T, typename K>
void fill_fast(std::vector<IsomerBatch<T,K>>& B){
  int N = B[0].n_atoms;
  int Nf = B[0].n_faces;
  int N_graphs = B[0].m_capacity;
  auto buckyqueue = BuckyGen::start(N, false, false);
  {
  auto face_degrees_acc = host_accessor(B[0].face_degrees);
  auto dual_neighbours_acc = host_accessor(B[0].dual_neighbours);
  auto statuses_acc = host_accessor(B[0].statuses);
  Graph G;

  G.neighbours = neighbours_t(Nf, std::vector<node_t>(6,-1));
  G.N = Nf;
  int num_generated = 0;
  bool more_isomers = true;
  //Stop at max 10000 graphs this shouldn't take too long
  for (int i = 0; (i < std::min(int(10000),N_graphs)) && (more_isomers = BuckyGen::next_fullerene(buckyqueue, G)); ++i) {
    num_generated++;
    statuses_acc[i] = IsomerStatus::NOT_CONVERGED;
    for(int j = 0; j < Nf; j++) {
      const auto &nj = G.neighbours[j];
      face_degrees_acc[i*Nf + j] = nj.size();
      for(int k = 0; k < G.neighbours[j].size(); k++) 
        dual_neighbours_acc[i*Nf*6 + j*6 + k] = nj[k];

    }
  }
  
  if (num_generated < N_graphs) 
    for (int i = num_generated; i < N_graphs; ++i) {
      statuses_acc[i] = IsomerStatus::NOT_CONVERGED;
      //Repeat the same graphs as already generated.
      for(int j = 0; j < Nf; j++) {
        face_degrees_acc[i*Nf + j] = face_degrees_acc[(i%num_generated)*Nf + j];
        for(int k = 0; k < 6; k++) 
          dual_neighbours_acc[i*Nf*6 + j*6 + k] = dual_neighbours_acc[(i%num_generated)*Nf*6 + j*6 + k];

      }
    }

  }

  for (int i = 1; i < B.size(); ++i) {
    copy(B[i], B[0]);
  }
  BuckyGen::stop(buckyqueue);
}

template void fill_fast(std::vector<IsomerBatch<float,uint16_t>>& B);

template void fill(IsomerBatch<float,uint16_t>& B, int set_div, int offset);
template void fill(IsomerBatch<double,uint16_t>& B, int set_div, int offset);

template void bucky_fill(IsomerBatch<float,uint16_t>& B, int ntasks, int mytask_id);
template void bucky_fill(IsomerBatch<double,uint16_t>& B, int ntasks, int mytask_id);

template bool bucky_fill(IsomerBatch<float,uint16_t>& B, BuckyGen::buckygen_queue &Q);
template bool bucky_fill(IsomerBatch<double,uint16_t>& B, BuckyGen::buckygen_queue &Q);
