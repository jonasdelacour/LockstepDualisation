#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include "cu_array.h"
#include "cuda_runtime.h"
#include <ostream>
#include <iostream>

template <typename T>
CuArray<T>::CuArray() : size_(0), capacity_(0){}

template <typename T>
CuArray<T>::CuArray(const size_t size) : size_(size), capacity_(size){
    cudaMallocManaged(&data, size*sizeof(T));
    //cudaMemSet(data,0, size*sizeof(T));
}

template <typename T>
CuArray<T>::CuArray(const CuArray<T>& other) : size_(other.size_), capacity_(other.capacity_){
    cudaMallocManaged(&data, size_*sizeof(T));
    std::cout << "Copy constructor called\n";
}

template <typename T>
void CuArray<T>::resize(const size_t size){
    T* new_ptr;
    cudaMallocManaged(&new_ptr, size*sizeof(T));
    cudaMemcpy(new_ptr, data, size_*sizeof(T), cudaMemcpyHostToHost);
    
    cudaFree(data);
    data = new_ptr;
}

template <typename T>
size_t CuArray<T>::size(){
    return size_;
}

template <typename T>
__device__ __host__ T& CuArray<T>::operator[] (const size_t i){
    return data[i];
}

template <typename T>
CuArray<T>& CuArray<T>::operator=(const CuArray<T>& other){
    if (this != &other){
        cudaMallocManaged(&data, other.size_*sizeof(T));
        size_ = other.size_;
        capacity_ = other.capacity_;
        cudaMemcpy(data, other.data, size_*sizeof(T), cudaMemcpyHostToHost );
    }
    return *this;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const CuArray<T>& input){
    os << "[" << input.data[0];
    for (size_t i = 1; i < input.size_; ++i){
        os << "," << input.data[i];
    }
    os << "]\n";
    return os;
}

template <typename T>
CuArray<T>::~CuArray(){
    cudaFree(data);
}



template <typename T>
void CuArray<T>::to_device(const int device){
    static int attr;
    static bool attr_set = false;
    if (!attr_set){
        cudaDeviceGetAttribute(&attr,cudaDevAttrMemoryPoolsSupported, device);
        attr_set = true;
    }
    if (attr) {
        cudaSetDevice(device);
        cudaMemPrefetchAsync(data, size_*sizeof(T), device);
        cudaDeviceSynchronize(); //Ensures that the data is copied before the function returns. Enabled for benchmarking purposes, remove for performance
    }
} 

template <typename T>
void CuArray<T>::to_host(const int device){
    static int attr;
    static bool attr_set = false;
    if (!attr_set){
        cudaDeviceGetAttribute(&attr,cudaDevAttrMemoryPoolsSupported, device);
        attr_set = true;
    }
    if (attr) {
        cudaSetDevice(device);
        cudaMemPrefetchAsync(data, size_*sizeof(T), cudaCpuDeviceId);
        cudaDeviceSynchronize(); //Ensures that the data is copied before the function returns. Enabled for benchmarking purposes, remove for performance
    }
}

//Primitive way of handling the fact that templated code in this translation unit wont be generated unless explicitly instantiated somewhere.
int declare_cu_arrays(){
    CuArray<float> f1_a;                    CuArray<float> f1(1);           f1_a = f1;f1[0] = {};   std::cout << f1;  f1.to_device(0); f1.to_host(0); 
    CuArray<double> f2_a;                   CuArray<double> f2(1);          f2_a = f2;f2[0] = {};   std::cout << f2; f2.to_device(0); f2.to_host(0);
    CuArray<int> f7_a;                      CuArray<int> f7(1);             f7_a = f7;f7[0] = {};   std::cout << f7; f7.to_device(0); f7.to_host(0);
    CuArray<size_t> f8_a;                   CuArray<size_t> f8(1);          f8_a = f8;f8[0] = {};   std::cout << f8; f8.to_device(0); f8.to_host(0);
    CuArray<uint8_t> f9_a;                  CuArray<uint8_t> f9(1);         f9_a = f9;f9[0] = {};   std::cout << f9; f9.to_device(0); f9.to_host(0);
    CuArray<unsigned char> f9_2_a;          CuArray<unsigned char> f9_2(1); f9_2_a = f9_2;f9_2[0] = {}; std::cout << f9_2; f9_2.to_device(0); f9_2.to_host(0);
    CuArray<uint16_t> f10_a;                CuArray<uint16_t> f10(1);       f10_a = f10;f10[0] = {};  std::cout << f10; f10.to_device(0); f10.to_host(0);
    CuArray<unsigned short> f10_2_a;        CuArray<unsigned short> f10_2(1); f10_2_a = f10_2;f10_2[0] = {};  std::cout << f10_2; f10_2.to_device(0); f10_2.to_host(0);
    CuArray<uint32_t> f11_a;                CuArray<uint32_t> f11(1);       f11_a = f11;f11[0] = {};  std::cout << f11; f11.to_device(0); f11.to_host(0);
    CuArray<uint64_t> f12_a;                CuArray<uint64_t> f12(1);       f12_a = f12;f12[0] = {};  std::cout << f12; f12.to_device(0); f12.to_host(0);
    CuArray<char> f13_a;                    CuArray<char> f13(1);           f13_a = f13;f13[0] = {};  std::cout << f13; f13.to_device(0); f13.to_host(0);
    CuArray<bool> f14_a;                    CuArray<bool> f14(1);           f14_a = f14;f14[0] = {};  std::cout << f14; f14.to_device(0); f14.to_host(0);
    return 1;
}
