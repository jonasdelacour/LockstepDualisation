#!/usr/bin/env python3

# %% Set up benchmark parameters
SYCL_setvars = '/opt/intel/oneapi/setvars.sh'
batch_size=2**15
gen_batch_size=3000000 
Ngpu_runs = 20 #Set to 100 for more accurate results, much smaller standard deviation.
Ncpu_runs = 20 #Set to 100 for more accurate results, much smaller standard deviation.
Ncpu_warmup = 1 #Warmup caches and branch predictor.
Ngpu_warmup = 1 #No branch prediction on GPU, but SYCL runtime incurs overhead the first time each kernel is run.
N_warmup = { "cpu": 5, "gpu": 50 }
N_runs = { "cpu": 20, "gpu": 10 }
#Change this number if the simulation is taking too long.
#Setting this number to -1 will reduce the batch sizes by 1 power of 2 in the kernel dualisation benchmark.
Bathcsize_Offset = { "cpu": -5, "gpu": 0 } 


#----------------- BENCHMARK DEFINITIONS --------------------
import numpy as np, pandas as pd
import os, subprocess, platform, pathlib, sys

# Benchmark result filenames
hostname = platform.node()
path = f'{os.getcwd()}/output/{hostname}/'
buildpath = f'{os.getcwd()}/build/'
pathlib.Path(f"{path}/figures").mkdir(parents=True, exist_ok=True)


if(len(sys.argv)<2):
    print(f"Syntax: {sys.argv[0]} <task>\n"+
          "Where task is one of: 'all', 'batchsize', 'baseline', 'dualize_gpu', 'dualize_cpu', 'generate', 'pipeline', or 'validate'.\n");
    exit(-1)
    
task = sys.argv[1]

# The benchmark output filenames are:
# f'{path}/base.csv' 
# f'{path}/one_gpu_v0.csv'    
# f'{path}/one_gpu_v1.csv'    
# f'{path}/multi_gpu_v0.csv'  
# f'{path}/multi_gpu_v1.csv'  
# f'{path}/multi_gpu_weak.csv'
# f'{path}/single_gpu_bs.csv' 
# f'{path}/single_gpu_bs.csv' 
# f'{path}/base_pipeline.csv' 
# f'{path}/full_pipeline.csv' 

# TODO: Use SYCL to make benchmark cross-vendor
if os.system('which nvidia-smi') == 0:
    output = subprocess.check_output(['nvidia-smi', '-L'])

    # Convert the byte string to a regular string
    output_str = output.decode('utf-8')

    # Count the number of lines in the output
    num_gpus = len(output_str.strip().split('\n'))
else:
    num_gpus = 0
# Print the number of GPUs found
print(f'Found {num_gpus} GPUs on {hostname}')


def source_and_get_environment(script_path, base_environment=None):
    """Source script and return the updated environment."""
    if base_environment is None:
        base_environment = os.environ.copy()
    command = ['/bin/bash', '-c', f'source {script_path} ; env']
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, env=base_environment)
    output, _ = proc.communicate()
    env = dict((line.split("=", 1) for line in output.decode().splitlines() if "=" in line))
    return env

def reset_file(filename,header=""):
    with open(filename, 'w', newline='') as f:
        if(header != ""): print(header,file=f)

env = source_and_get_environment(SYCL_setvars)


def validate_kernel():
# #Validate SYCL kernels against baseline dualisation
    if os.path.exists(f'{buildpath}validation/sycl/sycl_validation'):
        subprocess.Popen(['/bin/bash', '-c', f'{buildpath}validation/sycl/sycl_validation gpu'], env=env).wait()
    else:
        print("SYCL validation kernel not found. Make sure your SYCL environment is set up correctly. Then run `make all` again.")
        


def bench_batchsize():
# # ### Run the batch size experiment
    reset_file(f'{path}/single_gpu_bs.csv')
    if os.path.exists(f'{buildpath}benchmarks/sycl/dualisation') and num_gpus>0:
        for i in range(0,22):
            subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/dualisation gpu {200} {2**(i)} {N_runs["gpu"]} {N_warmup["gpu"]} 4 1 {path}/single_gpu_bs.csv'], env=env).wait()
    elif os.path.exists(f'{buildpath}benchmarks/sycl/dualisation'):
        print("No GPUs found. Skipping batch size experiment.")
    else:
        print("SYCL dualisation kernel not found. Make sure your SYCL environment is set up correctly. Then run `make all` again.")

        
# # ### Run the benchmarks

def bench_baseline():
    def check_file_exists(filepath, message):
        if not os.path.exists(filepath):
            print(message)
            return False
        return True

    # Usage example:
    if not check_file_exists(f'{buildpath}benchmarks/baseline', "Baseline binary not found. Skipping baseline benchmark"):
        return
    reset_file(f'{path}/base.csv')
    for i in range(20,201,2):
        os.system(f'{buildpath}benchmarks/baseline {i} {2**(8)} {N_runs["cpu"]} {N_warmup["cpu"]} 0 {path}/base.csv')
    
def bench_dualize(kernel_versions="all", devices="cpu"):
    if not os.path.exists(f'{buildpath}benchmarks/sycl/dualisation'):
        print("SYCL dualisation kernel not found. Skipping dualisation benchmark. Make sure your SYCL environment is set up correctly. Then run `make all` again.")
        return
    
    devices = devices.lower()
    kernel_range = range(1, 5) if kernel_versions == "all" else [int(kernel_versions)]
    device_range = ["cpu", "gpu"] if devices == "both" else [devices]

    if "gpu" in device_range and num_gpus==0:
        print("No GPUs found. Skipping dualisation benchmark.")
        return
    if "gpu" in device_range: 
        reset_file(f'{path}/multi_gpu_weak.csv')
    if "cpu" in device_range:
        #reset_file(f'{path}/omp_multicore_sm.csv')
        reset_file(f'{path}/omp_multicore_tp.csv')

    for j in kernel_range:
        if "gpu" in device_range: 
            reset_file(f'{path}/multi_gpu_v{j}.csv')
        for device in device_range:
            reset_file(f'{path}/one_{device}_v{j}.csv')
    for i in range(20,201,2):
        #Currently just running weak scaling for multi-GPU using the fastest kernel (v1)
        if "cpu" in device_range:
            #OpenMP Benchmark (2 Different Versions: Shared Memory Parallelism and Task Parallelism)
            #proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/omp_multicore {i} {2**(20+Bathcsize_Offset["cpu"])} {N_runs["cpu"]} {N_warmup["cpu"]} 0 {path}/omp_multicore_sm.csv'], env=env); proc.wait()
            proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/omp_multicore {i} {2**(20+Bathcsize_Offset["cpu"])} {N_runs["cpu"]} {N_warmup["cpu"]} 1 {path}/omp_multicore_tp.csv'], env=env); proc.wait()
        if "gpu" in device_range:
            proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/dualisation gpu {i} {num_gpus*2**(20+Bathcsize_Offset["gpu"])} {N_runs["gpu"]} {N_warmup["gpu"]} 1 {num_gpus} {path}/multi_gpu_weak.csv'], env=env); proc.wait()
        for j in kernel_range:
            if "gpu" in device_range:
                proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/dualisation gpu {i} {2**(20+Bathcsize_Offset["gpu"])} {N_runs["gpu"]} {N_warmup["gpu"]} {j} {num_gpus} {path}/multi_gpu_v{j}.csv'], env=env); proc.wait()
            for device in device_range:
                proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/dualisation {device} {i} {2**(20+Bathcsize_Offset[device])} {N_runs[device]} {N_warmup[device]} {j} 1 {path}/one_{device}_v{j}.csv'], env=env);  proc.wait()
            



def bench_generate():
    reset_file(f"{path}/buckybench.csv", header="N,BS,T_gen,TSD_gen")
    if not os.path.exists(f'{buildpath}benchmarks/buckybench'):
        print("Buckybench kernel not found. Skipping buckybench benchmark. Make sure requirements are all met and run `make all` again.")
        return
    for N in range(72,201,2):
        proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/buckybench {N} {gen_batch_size} 1 0 >>  {path}/buckybench.csv'], env=env).wait()
    

# Benchmarking breakdown of timings
def bench_pipeline():
    reset_file(f'{path}/base_pipeline.csv')
    reset_file(f'{path}/full_pipeline.csv')
    if not os.path.exists(f'{buildpath}benchmarks/sycl/baseline_pipeline'):
        print("SYCL baseline pipeline kernel not found. Skipping baseline pipeline benchmark. Make sure your SYCL environment is set up correctly. Then run `make all` again.")
        return
    if num_gpus==0:
        print("No GPUs found. Skipping pipeline benchmark.")
        return
    for N in range(72,201,2):
        proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/baseline_pipeline gpu {N} {batch_size} {N_runs["gpu"]} {N_warmup["gpu"]} 1 1 {path}/base_pipeline.csv'], env=env).wait()
        proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/pipeline gpu {N} {batch_size} {N_runs["gpu"]} {N_warmup["gpu"]} 1 1 {path}/full_pipeline.csv'], env=env).wait()




def bench_dualize_cpu():
    bench_dualize("all", "cpu")

def bench_dualize_gpu():
    bench_dualize("all", "gpu")


tasks = {'batchsize':   bench_batchsize,
         'baseline':    bench_baseline,
         'dualize_cpu': bench_dualize_cpu,
         'dualize_gpu': bench_dualize_gpu,
         'generate':    bench_generate,
         'pipeline':    bench_pipeline,
         'validate':    validate_kernel};

if(task=="all"):
    for k in tasks: tasks[k]()
else:
    tasks[task]()

