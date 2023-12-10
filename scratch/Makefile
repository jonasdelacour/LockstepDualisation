MPICXX=/usr/bin/mpic++
MPIRUN=/usr/bin/mpirun
HOSTS=-host threadripper00:32,threadripper01:32
PWD=$(shell pwd)
BASEDIR=$(shell dirname $$PWD)

CUDA_LIB_PATH=$(PWD)/build/src/cuda
CUDA_LIB=-L$(CUDA_LIB_PATH) -lcuda_lib
CPP_LIB=-L$(PWD)/build/src/c++ -lcpp_lib
INCLUDES=-I$(PWD)/include/

all: build run

build: mpi_hello_world.cpp
	$(MPICXX) -o mpi_hello_world mpi_hello_world.cpp $(INCLUDES) $(CUDA_LIB) -I /opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/11.8/include/
	rsync -r $(PWD) threadripper01:$(BASEDIR)

run: mpi_hello_world host_file
	$(MPIRUN) -hostfile host_file $(HOSTS) -mca btl_tcp_if_include enp8s0 -mca btl_tcp_if_exclude,lo docker0 -N 1 $(PWD)/mpi_hello_world 200 1000000 20 1 1 

clean:
	rm -f mpi_hello_world