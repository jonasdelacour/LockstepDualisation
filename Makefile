output_dir = output/$(shell hostname)
PYTHON=python3

.PHONY: benchmark_single_node configure binaries validation plot_benchmarks

all: configure binaries benchmark_single_node

configure:
	${PYTHON} -m pip install -r requirements.txt
	mkdir -p build
	mkdir -p ${output_dir}
	cd build ; ccmake .. -DCMAKE_BUILD_TYPE=Release ; cd -

build/CMakeCache.txt: configure

binaries: build/CMakeCache.txt
	cd build ; make -j ; cd -

build/validation/sycl/sycl_validation: binaries

validation: ./build/validation/sycl/sycl_validation
	./build/validation/sycl/sycl_validation cpu | tee ${output_dir}/validation_cpu.log
	./build/validation/sycl/sycl_validation gpu | tee ${output_dir}/validation_gpu.log

benchmark_single_node: configure binaries
	${PYTHON} benchmarks.py all

plot_benchmarks:
	${PYTHON} plot-benchmarks.py 




