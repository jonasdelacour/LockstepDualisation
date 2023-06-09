ifeq ($(shell which nvcc),) # No 'nvcc' found
HAS_GPU=false
else
HAS_GPU=true
endif

.PHONY: build
build:
	git submodule update --init
	mkdir -p build
	cd build && cmake .. && make -j

.PHONY: all
all: build benchmarks validation

.PHONY: benchmarks
benchmarks: build
	python3 benchmarks.py

.PHONY: validation
ifeq ($(HAS_GPU), true)
validation: build
	build/validation/omp_validation
	build/validation/gpu_validation
else
validation: build
	build/validation/omp_validation
endif

.PHONY: clean
clean:
	rm -rf build/