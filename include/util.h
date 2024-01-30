#pragma once
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <string>
#include "fullerenes/polyhedron.hh"
#include "fullerenes/buckygen-wrapper.hh"
#include "fullerenes/isomerdb.hh"

template <typename T>
T mean(const std::vector<T>& v);

template<typename T>
T stddev(const std::vector<T>& data);

template<typename T>
void remove_outliers(std::vector<T>& data, int n_sigma = 2);

#include "isomer_batch.h"

template <typename T, typename K>
void fill(IsomerBatch<T,K>& B, int set_div = 1,int offset = 0);

template <typename T, typename K>
void bucky_fill(IsomerBatch<T,K>& B, int mytask_id, int ntasks);

template <typename T, typename K>
bool bucky_fill(IsomerBatch<T,K>& B, BuckyGen::buckygen_queue& BuckyQ);


size_t filesize(std::ifstream &f);

std::string cwd();
