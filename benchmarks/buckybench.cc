#include <fullerenes/buckygen-wrapper.hh>
#include <fullerenes/triangulation.hh>
#include <fullerenes/symmetry.hh>
#include <fullerenes/isomerdb.hh>
#include <random>
#include <chrono>
#include <unistd.h>

using namespace std::chrono;
using namespace std;

double mean(const std::vector<double>& v) {
  double sum = 0;
  for(size_t i=0;i<v.size();i++) sum += v[i];
  return sum/v.size();
}

double stddev(const vector<double>& data)
{
  if(data.size() <= 1) return 0;

  double mn = mean(data);

  double sum_of_squares = 0.0;
  for (const double& value : data){
    double diff = value - mn;
    sum_of_squares += diff * diff;
  }

  double variance = sum_of_squares / (data.size() - 1);
  return std::sqrt(variance);
}


int main(int ac, char **av)
{
  size_t default_M = 1000000, default_batches = 2;
  if(ac<2) {
    fprintf(stderr,"Syntax: %s <N> [M:%ld] [n_batches:%ld] [my_batch:0] [IPR:0] [only_symmetric:0]\n"
	    "\tN: Number of C atoms (C_N)\n"
	    "\tM: Total number of isomers to generate and benchmark\n"
	    "\tn_batches: Number of batches\n"
	    "\tmy_batch:  Which batch to generate\n"
	    "\tseed: Random seed for reproducibility. Use -1 to seed with random device for multiple different randomizations.\n"
	    "\tIPR:  If 1, only generate IPR-fullerenes\n"
	    "\tonly_symmetric: If 1, only generate symmetric fullerenes (>C1)\n\n",
	    av[0], default_M, default_batches);
    return -1;
  }
  bool
    IPR = ac>5? strtol(av[5],0,0):0,
    only_symmetric = ac>6? strtol(av[6],0,0):0;

  int     N = strtol(av[1],0,0);  
  int64_t Nisomers = IsomerDB::number_isomers(N,only_symmetric?"Nontrivial":"Any",IPR);
  size_t
    M         = ac>2? strtol(av[2],0,0):default_M,
    n_batches = ac>3? strtol(av[3],0,0):default_batches,
    my_batch  = ac>4? strtol(av[4],0,0):0;
    M = ::min<size_t>(M,Nisomers/n_batches);
  

  // Benchmark Buckygen graph generation without payload
  steady_clock::time_point t0, t1;
  using ns_duration = duration<double,std::nano>;

  BuckyGen::buckygen_queue BQ = BuckyGen::start(N,IPR,only_symmetric,my_batch,n_batches);
  Triangulation g;
  
  size_t isomer_ix = 0;

  vector<double> bucky_times(M);
  
  t0 = steady_clock::now(); 
  while( BuckyGen::next_fullerene(BQ,g) && (isomer_ix < M)){
    t1 = steady_clock::now();
    bucky_times[isomer_ix] = ns_duration(t1-t0).count();
    isomer_ix++;        
    t0 = steady_clock::now();
  }
  BuckyGen::stop(BQ);
  //N,BS,T_gen,TSD_gen
  printf("%d, %ld, %g, %g\n", N, M, mean(bucky_times), stddev(bucky_times));  
  
  return 0;
}
