output_dir = output/$(shell hostname)
PYTHON=python3

configure:
	mkdir -p build
	cd build ; ccmake .. -DCMAKE_BUILD_TYPE=Release ; cd -

build/CmakeCache.txt: configure

binaries: build/CMakeCache.txt
	cd build ; make -j ; cd -

build/validation/sycl/sycl_validation: binaries

validation: ./build/validation/sycl/sycl_validation
	mkdir -p ${output_dir}
	./build/validation/sycl/sycl_validation cpu | tee ${output_dir}/validation_cpu.log
	./build/validation/sycl/sycl_validation gpu | tee ${output_dir}/validation_gpu.log


benchmarks_single_node: configure binaries
	$PYTHON -m pip install -r requirements.txt
	$PYTHON benchmarks.py




