# %% [markdown]
# ### Setup

# %%
import numpy as np, pandas as pd
import os, subprocess, platform, pathlib

Ngpu_runs = 10 #Set to 100 for more accurate results, much smaller standard deviation.
Ncpu_runs = 10 #Set to 100 for more accurate results, much smaller standard deviation.
Ncpu_warmup = 5 #Warmup caches and branch predictor.
Ngpu_warmup = 1 #No branch prediction on GPU, but if prefetching is disabled, the first run will fetch data.
#Change this number if the simulation is taking too long.
#Setting this number to -1 will reduce the batch sizes by 1 power of 2
OFFSET_BS = 0

# Benchmark result filenames
hostname = platform.node()
path = f'{os.getcwd()}/output/{hostname}/'
buildpath = f'{os.getcwd()}/build/'
pathlib.Path(f"{path}/figures").mkdir(parents=True, exist_ok=True)

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

# Run the command and capture its output
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

env = source_and_get_environment('/opt/intel/oneapi/setvars.sh')

# %%
def reset_file(filename):
    # Check if the file already exists
    if os.path.isfile(filename):
        # If it does, empty it by opening it in write mode with the 'truncate' option
        with open(filename, 'w', newline='') as f:
            f.truncate()
    else:
        # If it doesn't, create an empty file by opening it in write mode
        with open(filename, 'w', newline='') as f:
            pass


# # TODO: Only delete results when re-running benchmark
reset_file(f'{path}/base.csv')
reset_file(f'{path}/one_gpu_v0.csv')
reset_file(f'{path}/one_gpu_v1.csv')
reset_file(f'{path}/multi_gpu_v0.csv')
reset_file(f'{path}/multi_gpu_v1.csv')
reset_file(f'{path}/multi_gpu_weak.csv')

# %% [markdown]
# ### Validation

# %%
#Validate SYCL kernels against baseline dualisation

process = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}validation/sycl/sycl_validation gpu'], env=env).wait()

# %% [markdown]
# ### Run the batch size experiment

# %%
reset_file(f'{path}/single_gpu_bs.csv')


for i in range(0,20):
    if(num_gpus>0):
        proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/dualisation gpu {200} {2**(i)} {Ngpu_runs} {Ngpu_warmup} 1 1 {path}/single_gpu_bs.csv'], env=env); proc.wait()


        
# %% [markdown]
# ### Run the benchmarks

# %%
reset_file(f'{path}/base.csv')
reset_file(f'{path}/one_gpu_v0.csv')
reset_file(f'{path}/one_gpu_v1.csv')
reset_file(f'{path}/multi_gpu_v0.csv')
reset_file(f'{path}/multi_gpu_v1.csv')
reset_file(f'{path}/multi_gpu_weak.csv')

for i in range(20,201,2):
    os.system(f'{buildpath}benchmarks/baseline {i} {2**(6+OFFSET_BS)} {Ncpu_runs} {Ncpu_warmup} 0 {path}/base.csv')
    if(num_gpus>0):
        proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/dualisation gpu {i} {2**(20+OFFSET_BS)} {Ngpu_runs} {Ngpu_warmup} 0 1 {path}/one_gpu_v0.csv'], env=env);  proc.wait()
        proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/dualisation gpu {i} {2**(20+OFFSET_BS)} {Ngpu_runs} {Ngpu_warmup} 1 1 {path}/one_gpu_v1.csv'], env=env); proc.wait()
        proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/dualisation gpu {i} {2**(20+OFFSET_BS)} {Ngpu_runs} {Ngpu_warmup} 0 {num_gpus} {path}/multi_gpu_v0.csv'], env=env); proc.wait()
        proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/dualisation gpu {i} {2**(20+OFFSET_BS)} {Ngpu_runs} {Ngpu_warmup} 1 {num_gpus} {path}/multi_gpu_v1.csv'], env=env); proc.wait()
        proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/dualisation gpu {i} {num_gpus*2**(20+OFFSET_BS)} {Ngpu_runs} {Ngpu_warmup} 1 {num_gpus} {path}/multi_gpu_weak.csv'], env=env); proc.wait()


    

# %%

#Benchmarking full pipeline with sequential dualisation
reset_file(f'{path}/base_pipeline.csv')
reset_file(f'{path}/full_pipeline.csv')
batch_size=10000
for N in range(72,201,2):
    proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/baseline_pipeline gpu {N} {batch_size} {Ncpu_runs} {Ngpu_warmup} 1 1 {path}/base_pipeline.csv'], env=env).wait()
    proc = subprocess.Popen(['/bin/bash', '-c', f'{buildpath}benchmarks/sycl/pipeline gpu {N} {batch_size} {Ngpu_runs} {Ngpu_warmup} 1 1 {path}/full_pipeline.csv'], env=env).wait()






