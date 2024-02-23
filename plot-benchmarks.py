import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, colorsys
from matplotlib import rcParams as rc
import os, sys,subprocess, platform
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset)
from os.path import relpath

Fontsize = 30
rc["legend.markerscale"] = 2.0
rc["legend.framealpha"] = 0
rc["legend.labelspacing"] = 0.1
rc['figure.figsize'] = (20,10)
rc['axes.autolimit_mode'] = 'data'
rc['axes.xmargin'] = 0
rc['axes.ymargin'] = 0.10
rc['axes.titlesize'] = 30
rc['axes.labelsize'] = Fontsize
rc['xtick.direction'] = 'in'
rc['ytick.direction'] = 'in'
rc['font.sans-serif'] = "Times New Roman"
rc['font.serif'] = "Times New Roman"
rc['xtick.labelsize'] = Fontsize
rc['ytick.labelsize'] = Fontsize
rc['axes.grid'] = True
rc['grid.linestyle'] = '-'
rc['grid.alpha'] = 0.2
rc['legend.fontsize'] = int(Fontsize*0.9)
rc['legend.loc'] = 'upper left'
rc["figure.autolayout"] = True
rc["savefig.dpi"] = 300
rc["text.usetex"] = True
rc["font.family"] = "Times New Roman"
rc["lines.markeredgecolor"] = matplotlib.colors.to_rgba('black', 0.5)
rc["lines.markeredgewidth"] = 0.01
rc["legend.markerscale"] = 2.0
rc['text.latex.preamble'] = r'\usepackage{amssymb}'
rc.update({
  "text.usetex": True,
  "font.family": "Times New Roman"
})

def set_fontsizes(fontsize):
  rc['axes.labelsize'] = fontsize
  rc['xtick.labelsize'] = fontsize
  rc['ytick.labelsize'] = fontsize
  rc['legend.fontsize'] = int(fontsize*0.9)

#Lets use the following markers: Triangle, Square, Circle, Star, Diamond
Markers = ['o','^', 's', '*', 'D']
MarkerScales = np.array([1.1, 1.25, 1., 1.5, 1.])


def read_csv(fname):
  try:
    df = pd.read_csv(fname)
    return df["N"].to_numpy(), df["T"].to_numpy(), df["TSD"].to_numpy()
  except Exception as e:
    print(f"Could not read {fname}, skipping due to {e}")
    return None, None, None

#Color dictionary
CD = { 
  "Baseline" : 'r', 
  "CPU_V1" :    "#FDEE00", 
  "CPU_V2" :    "#06D6A0", 
  "CPU_V3" :    "#FF4365", 
  "CPU_V4" :    "#14080E", 
  "OMP_TP" :    "#6320EE",
  "OMP_SM" :    "#963D5A", 

  "Dual" :      "#7570b3", 
  "Generate" :  "#d95f02", 
  "Projection" :"#e7298a", 
  "Tutte" :     "#66a61e", 
  "Opt" :       "#8931EF", 

  "GPU_V1" :    "#1f77b4", 
  "GPU_V2" :    "#e377c2", 
  "GPU_V3" :    "#0D9276",
  "GPU_V4" :    "#8c564b",
}

if(len(sys.argv)>1):
    benchname = sys.argv[1]
else:
    benchname = platform.node()

# Benchmark result filenames
cwd = os.getcwd()
path = f'/{cwd}/output/{benchname}/'
buildpath = f'/{cwd}/build/'
fname_base = f'{path}base.csv'
fname_one_gpu_dual = f'{path}one_gpu_v'
fname_multi_gpu_dual = f'{path}multi_gpu_v'
fname_multi_gpu_weak = f'{path}multi_gpu_weak.csv'
fname_single_gpu_bs = f'{path}single_gpu_bs.csv'
fname_single_gpu_bs = f'{path}single_gpu_bs.csv'
fname_base_pipeline = f'{path}base_pipeline.csv'
fname_full_pipeline = f'{path}full_pipeline.csv'
fname_omp = f'{path}omp_multicore_'
fname_one_cpu = f'{path}one_cpu_v' 
save_format = "pdf"
KName = r"SYCL "

os.makedirs(f"{path}/figures",exist_ok=True)

def plot_kernel_cpu_benchmarks():
  print(f"Plotting batch size benchmark from {relpath(fname_multi_gpu_dual + '1.csv',cwd)} to {relpath(path,cwd)}/figures/kernel_benchmark.pdf")
  MS = MarkerScales * rc["lines.markersize"]
  fig, ax = plt.subplots(figsize=(15,15), nrows=2, sharex=False, dpi=200)
  N, T, TSD = read_csv(fname_omp + "tp.csv")
  if N is not None:
  
    #ax[0].fill_between(N, (T - TSD*2), (T+TSD*2), color='k', alpha=0.1, label=r"2$\sigma$")
    #ax[0].plot(N, T, f'{Markers[-1]}:', ms=MS[-1], color=CD["OMP_TP"], label=KName + r"OpenMP [CPU]" + " Task-Parallel")
    ax[0].fill_between(N, (T - TSD*2) / N, (T+TSD*2) / N, color='k', alpha=0.1, label=r"2$\sigma$") 
    ax[0].plot(N, T / N, f'{Markers[-1]}:', ms=MS[-1], color=CD["OMP_TP"], label= r"OpenMP [CPU]" + " Task-Parallel")

  N, T, TSD = read_csv(fname_omp + "tp_scaling.csv")
  if N is not None:
    Ncores = range(1,T.shape[0]+1)
    ax[1].plot(Ncores, Ncores, color="r", ls='--', label= r"Linear Scaling" )
    ax[1].fill_between(Ncores, T[0]/(T - TSD*2), T[0]/(T+TSD*2), color='k', alpha=0.1, label=r"2$\sigma$")
    ax[1].plot(Ncores, T[0]/T, f'{Markers[-1]}:', ms=MS[-1], color=CD["OMP_TP"], label= r"OpenMP [CPU] Task-Parallel")
  

  for i in [1,2,3,4]:
    fname = fname_one_cpu + str(i) + ".csv"
    N, T, TSD = read_csv(fname)
    if N is None: continue
    ax[0].fill_between(N, (T - TSD*2) / N, (T+TSD*2) / N, color='k', alpha=0.1)
    ax[0].plot(N, T / N, f'{Markers[i-1]}:', ms=MS[i-1], color=CD[f"CPU_V" + str(i)], label=KName + f" [CPU] V" + str(i))
    _, T, TSD = read_csv(fname_one_cpu + str(i) + "_scaling.csv")
    Ncores = range(1,T.shape[0]+1)
    if N is None: continue
    ax[1].fill_between(Ncores, T[0]/(T - TSD*2), T[0]/(T+TSD*2), color='k', alpha=0.1)
    ax[1].plot(Ncores, T[0]/T, f'{Markers[i-1]}:', ms=MS[i-1], color=CD[f"CPU_V" + str(i)], label=KName + f" [CPU] V" + str(i))


  ax[0].set_ylim(0,)
  ax[0].set_xlabel(r"Cubic Graph Size [\# Vertices]")
  ax[0].set_ylabel(r"Time / Vertex [ns]")
  ax[1].set_ylim(None, ax[1].get_ylim()[1] * 1.15)
  ax[1].legend(loc="upper left", ncol=2)
  ax[1].set_ylabel(r"Speedup")
  ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
  ax[1].set_xticks(range(2,os.cpu_count()+1,2))
  ax[1].set_xticklabels([f"{i}" for i in range(2,os.cpu_count()+1,2)])
  ax[1].set_xlabel(r"Number of Threads")
  plt.savefig(f"{path}/figures/kernel_benchmark_CPU.{save_format}", bbox_inches='tight')

def plot_kernel_gpu_benchmarks():
  print(f"Plotting batch size benchmark from {relpath(fname_multi_gpu_dual + '1.csv',cwd)} to {relpath(path,cwd)}/figures/kernel_benchmark.pdf")
  MS = MarkerScales * rc["lines.markersize"]
  fig, ax = plt.subplots(figsize=(15,15), nrows=2, sharex=True, dpi=200)
    
  for i in range(1,5):
    fname = fname_multi_gpu_dual + str(i) + ".csv"
    N, T, TSD = read_csv(fname)
    if N is None: continue
    if i == 1:
      ax[0].fill_between(N, (T - TSD*2), (T+TSD*2), color='k', alpha=0.1, label=r"2$\sigma$")
      ax[1].fill_between(N, (T - TSD*2)*1e3 / N, (T+TSD*2)*1e3 / N, color='k', alpha=0.1, label=r"2$\sigma$") 
    else:
      ax[0].fill_between(N, (T - TSD*2), (T+TSD*2), color='k', alpha=0.1)
      ax[1].fill_between(N, (T - TSD*2)*1e3 / N, (T+TSD*2)*1e3 / N, color='k', alpha=0.1)
    ax[0].plot(N, T, f'{Markers[i-1]}:', ms=MS[i-1], color=CD[f"GPU_V" + str(i)], label=KName + f" [GPU] V" + str(i))
    ax[1].plot(N, T*1e3 / N, f'{Markers[i-1]}:', ms=MS[i-1], color=CD[f"GPU_V" + str(i)], label=KName + f" [GPU] V" + str(i))
      
  ax[0].set_ylabel(r"Time / Graph [ns]")
  ax[0].set_ymargin(0.0)
  ax[1].set_ymargin(0.0)
  ax[0].vlines(96, ax[0].get_ylim()[0], ax[0].get_ylim()[1], color=CD["GPU_V4"], ls='--', label=r"Kernel 4 Saturation")
  ax[0].vlines(188, ax[0].get_ylim()[0], ax[0].get_ylim()[1], color=CD["GPU_V1"], ls='-.', label=r"Kernel 1 Saturation")
  ax[1].vlines(96, ax[1].get_ylim()[0], ax[1].get_ylim()[1], color=CD["GPU_V4"], ls='--', label=r"Kernel 4 Saturation")
  ax[1].vlines(188, ax[1].get_ylim()[0], ax[1].get_ylim()[1], color=CD["GPU_V1"], ls='-.', label=r"Kernel 1 Saturation")

  ax[0].legend(loc="upper left")
  ax[1].legend(bbox_to_anchor=(0.5, 0.9))
  ax[1].set_ylabel(r"Time / Vertex [ps]")
    
  ax[1].set_xlabel(r"Cubic Graph Size [\# Vertices]")
  plt.savefig(f"{path}/figures/kernel_benchmark_GPU.{save_format}", bbox_inches='tight')


## Batch size

def plot_batch_size():
  print(f"Plotting batch size benchmark from {relpath(fname_single_gpu_bs,cwd)} to {relpath(path,cwd)}/figures/batch_size_benchmark.{save_format}")
  MS = MarkerScales * rc["lines.markersize"]
  try:
    df_single_gpu_bs = pd.read_csv(fname_single_gpu_bs)
  except:
    print(f"Could not read {fname_single_gpu_bs}, skipping")
    return
  Nrows = df_single_gpu_bs.shape[0]
  fig,ax = plt.subplots(figsize=(15,10))
  def add_line(ax, BS, T, SD, label, color, marker, linestyle, ms_scale=1.):
      ax.plot(BS, T, marker=marker, color=color, label=label, ls=linestyle, ms = ms_scale*rc["lines.markersize"])
      ax.fill_between(BS, T - SD, T + SD, alpha=0.1, color='k')
  
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_ylabel("Time / Graph [ns]")
  ax.set_xlabel("Batch Size")
  add_line(ax, df_single_gpu_bs["BS"].to_numpy(), df_single_gpu_bs["T"].to_numpy(), df_single_gpu_bs["TSD"].to_numpy(), "Lockstep Parallel Dualization", CD["GPU_V2"], f'{Markers[0]}', ':', MarkerScales[0])
  ax.legend(loc='best')

  #Set xticks to powers of 2
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.NullFormatter())
  ax.xaxis.set_minor_locator(ticker.NullLocator())
  ax.xaxis.set_ticks([2**i for i in range(0,Nrows)])
  #Labels should be 2^i
  ax.xaxis.set_ticklabels([f'$2^{{{i}}}$' for i in range(0,Nrows)])
  ax.grid(True, which="both", ls="--", alpha=0.2)
  #remove the ticks from the bottom edge
  ax.tick_params(axis='x', which='both', bottom=False)

  #Create inset axes zooming in on the lower right corner
  axins = ax.inset_axes([0.5, 0.5, 0.4, 0.3])
  axins.set_yscale('log')
  axins.set_xscale('log')
  axins.set_ylabel("Time / Graph [ns]")
  axins.set_xlabel("Batch Size")
  #Add line for the last 5 data points
  add_line(axins, df_single_gpu_bs["BS"].to_numpy()[-7:], df_single_gpu_bs["T"].to_numpy()[-7:], df_single_gpu_bs["TSD"].to_numpy()[-7:], "Lockstep Parallel Dualization", CD["GPU_V2"], f'{Markers[0]}', ':', MarkerScales[0])
  axins.xaxis.set_major_locator(MaxNLocator(integer=True))
  axins.xaxis.set_major_formatter(ticker.ScalarFormatter())
  axins.xaxis.set_minor_formatter(ticker.NullFormatter())
  axins.xaxis.set_minor_locator(ticker.NullLocator())
  axins.yaxis.set_major_locator(MaxNLocator(integer=True))
  axins.yaxis.set_major_formatter(ticker.ScalarFormatter())
  axins.yaxis.set_minor_formatter(ticker.NullFormatter())
  axins.xaxis.set_ticks([2**i for i in range(Nrows-7,Nrows)])
  #Labels should be 2^i
  axins.xaxis.set_ticklabels([f'$2^{{{i}}}$' for i in range(Nrows-7,Nrows)])
  axins.grid(True, which="both", ls="--", alpha=0.2)
  #remove the ticks from the bottom edge
  axins.tick_params(axis='x', which='both', bottom=False)
  #Mark the zoomed area
  mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")

  plt.savefig(f'{path}figures/batch_size_benchmark.{save_format}', bbox_inches='tight')


## Baseline Sequential Dualization
def plot_baseline():
  MS = MarkerScales * rc["lines.markersize"]
  print(f"Plotting baseline benchmark from {relpath(fname_base,cwd)} to {relpath(path,cwd)}/figures/baseline.{save_format}")    
  #df_base = pd.read_csv(fname_base)
  N, T, TSD = read_csv(fname_base)
  if N is None: return

  fig, ax = plt.subplots(figsize=(15,15), nrows=2, sharex=True, dpi=200)
  ax[0].plot(N, T/1e3, f'{Markers[0]}:', ms=MS[0], color=CD["Baseline"], label="Baseline Sequential Dualisation")
  ax[0].fill_between(N, (T - TSD)/1e3, (T + TSD)/1e3, alpha=0.1, color='k')
  ax[0].set_ylabel(r"Time / Graph [$\mu$s]") #Times are converted to microseconds through division by 1e3
  ax[0].legend()

  ax[1].plot(N, T / N, f'{Markers[0]}:', ms=MS[0], color=CD["Baseline"], label="Baseline Sequential Dualisation")
  ax[1].fill_between(N, (T - TSD*2) / N, (T+TSD*2) / N, color='k', alpha=0.1, label=r"2$\sigma$")
  ax[1].set_ylabel(r"Time / Vertex [ns]")
  ax[1].set_xlabel(r"Cubic Graph Size [\# Vertices]")
  ax[1].legend()
  plt.savefig(path + f"figures/baseline.{save_format}", bbox_inches='tight')

##

def plot_weak_scaling():
  MS = MarkerScales * rc["lines.markersize"]
  print(f"Plotting scaling benchmark to {relpath(path,cwd)}/figures/dual_gpu_scaling.{save_format}")
  d1N, d1T, d1TSD = read_csv(fname_one_gpu_dual + "1.csv")
  d3N, d3T, d3TSD = read_csv(fname_multi_gpu_dual + "1.csv")
  d2N, d2T, d2TSD = read_csv(fname_multi_gpu_weak)
  if d1N is None or d3N is None or d2N is None: return

  def std_div(a,b, a_std, b_std):
      return a/b * np.sqrt((a_std/a)**2 + (b_std/b)**2)
  
  fig, ax     = plt.subplots(figsize=(15, 15), nrows=2, sharex=True)
  ax[0].plot(d1N, d1T, f'{Markers[0]}:', ms=MS[0],  color="k", label=f"1 GPU, $B_s = 2^{{{20}}}$")
  ax[0].plot(d3N, d3T, f'{Markers[1]}:', ms=MS[1],  color=CD["GPU_V1"], label=f"2 GPUs $B_s = 2^{{{21}}}$")
  ax[0].plot(d2N, d2T, f'{Markers[2]}--',ms=MS[2],  color=CD["GPU_V1"], label=f"2 GPUs $B_s = 2^{{{20}}}$")
  ax[0].fill_between(d1N, d1T - d1TSD*1, d1T + d1TSD*1, alpha=0.1, color='k', label=r"1$\sigma$")
  ax[0].fill_between(d3N, d3T - d3TSD*1, d3T + d3TSD*1, alpha=0.1, color='k')
  ax[0].fill_between(d2N, d2T - d3TSD*1, d2T + d3TSD*1, alpha=0.1, color='k')
  ax[0].set_ylabel("Time / Graph [ns]")
  ax[0].legend(loc='upper left')

  #Plot speedup
  ax[1].plot(d1N, d1T/d3T, f'{Markers[1]}:', ms=MS[1],  color=CD["GPU_V1"], label=f"2 GPUs $B_s = 2^{{{21}}}$")
  ax[1].plot(d1N, d1T/d2T, f'{Markers[2]}--',ms=MS[2],  color=CD["GPU_V1"], label=f"2 GPUs $B_s = 2^{{{20}}}$")
  std_1 = std_div(d1T, d3T, d1TSD, d3TSD)
  std_2 = std_div(d1T, d2T, d1TSD, d3TSD)
  ax[1].fill_between(d1N, d1T/d3T - std_1, d1T/d3T + std_1, alpha=0.1, color='k', label=r"1$\sigma$")
  ax[1].fill_between(d1N, d1T/d2T - std_2, d1T/d2T + std_2, alpha=0.1, color='k')
  ax[1].hlines(2, 20, 200, linestyles='dashed', color='k', label=r"Perfect Scaling")
  ax[1].set_ylabel("Speedup")
  ax[1].set_xlabel(r"Cubic Graph Size [\# Vertices]")
  ax[1].set_ylim(0.95,2*1.05)
  ax[1].legend(loc='lower right', ncol=2)
  plt.savefig(path + f"figures/dual_gpu_scaling.{save_format}", bbox_inches='tight')

def plot_pipeline(normalize=False):
  MS = MarkerScales * rc["lines.markersize"]
  print(f"Plotting pipeline benchmark from {relpath(fname_base_pipeline,cwd)} to {relpath(path,cwd)}/figures/pipeline.{save_format}")
  df_base_pipeline = pd.read_csv(fname_base_pipeline)

  fig, ax = plt.subplots(figsize=(20,10), nrows=1, sharex=True)

  opt = df_base_pipeline["T_opt"].to_numpy()
  opt_sd = df_base_pipeline["TSD_opt"].to_numpy()
  tutte = df_base_pipeline["T_tutte"].to_numpy()
  tutte_sd = df_base_pipeline["TSD_tutte"].to_numpy()
  project = df_base_pipeline["T_project"].to_numpy()
  project_sd = df_base_pipeline["TSD_project"].to_numpy()
  overhead = df_base_pipeline["T_overhead"].to_numpy()
  overhead_sd = df_base_pipeline["TSD_overhead"].to_numpy()
  gen = df_base_pipeline["T_gen"].to_numpy()
  gen_sd = df_base_pipeline["TSD_gen"].to_numpy()
  dual = df_base_pipeline["T_dual"].to_numpy()
  dual_sd = df_base_pipeline["TSD_dual"].to_numpy()
  natoms = df_base_pipeline["N"].to_numpy()


  parallel = opt + tutte + project
  parallel_sd = np.sqrt(opt_sd**2 + tutte_sd**2 + project_sd**2)
  total = parallel + overhead + gen + dual

  def plot_normalized_line(ax, x, y, y_sd, label, color, marker, linestyle, ms_scale=1.):
      ax.plot(x, 1e2* y/total, marker=marker, color=color, label=label, ls=linestyle, mfc=color, ms = ms_scale*rc["lines.markersize"]) #Normalized to total time, shown as percentage
      ax.fill_between(x, 1e2*(y - y_sd)/total, 1e2*(y + y_sd)/total, alpha=0.1, color='k')

  def plot_absolute_line(ax, x, y, y_sd, label, color, marker, linestyle, ms_scale=1.):
    ax.plot(x,  y/1e3, marker=marker, color=color, label=label, ls=linestyle, mfc=color, ms = ms_scale*rc["lines.markersize"]) #Normalized to total time, shown as percentage
    ax.fill_between(x, (y - y_sd)/1e3, (y + y_sd)/1e3, alpha=0.1, color='k')
  if normalize:
    plot_normalized_line(ax, natoms, gen, gen_sd, "Isomer-space graph generation", CD["Generate"], f'{Markers[0]}', ':', MarkerScales[0])
    plot_normalized_line(ax, natoms, parallel, parallel_sd, "Lockstep-parallel geometry optimization", "k", f'{Markers[1]}', ':', MarkerScales[1])
    plot_normalized_line(ax, natoms, overhead, overhead_sd, "Overhead", "blue", f'{Markers[2]}', ':', MarkerScales[2])
    plot_normalized_line(ax, natoms, dual, dual_sd, "Baseline sequential dualization", CD["Dual"], f'{Markers[3]}', ':', MarkerScales[3])
  else:
    plot_absolute_line(ax, natoms, gen, gen_sd, "Isomer-space graph generation", CD["Generate"], f'{Markers[0]}', ':', MarkerScales[0])
    plot_absolute_line(ax, natoms, parallel, parallel_sd, "Lockstep-parallel geometry optimization", "k", f'{Markers[1]}', ':', MarkerScales[1])
    plot_absolute_line(ax, natoms, overhead, overhead_sd, "Overhead", "blue", f'{Markers[2]}', ':', MarkerScales[2])
    plot_absolute_line(ax, natoms, dual, dual_sd, "Baseline sequential dualization", CD["Dual"], f'{Markers[3]}', ':', MarkerScales[3])
      
  ax.legend()
  ax.set_ylabel(r"Runtime Fraction [$\%$]") if normalize else ax.set_ylabel(r"Time / Graph [$\mu$s]")
  ax.set_xlabel(r"Isomerspace $C_N$")
  if normalize:
    ax.set_ylim(0,100)
  #percentage formatting
  ax.yaxis.set_major_formatter(ticker.PercentFormatter()) if normalize else ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
  normalized_str = "" if not normalize else "_normalized"
  plt.savefig(path + "figures/pipeline" + normalized_str + f".{save_format}", bbox_inches='tight')

def plot_lockstep_pipeline(normalize=False, log=False):
  MS = MarkerScales * rc["lines.markersize"]
  print(f"Plotting lockstep pipeline benchmark from {relpath(fname_full_pipeline,cwd)} to {relpath(path,cwd)}/figures/lockstep_pipeline.{save_format}")
  df_full_pipeline = pd.read_csv(fname_full_pipeline)

  fig, ax = plt.subplots(figsize=(20,10), nrows=1, sharex=True)

  opt         = df_full_pipeline["T_opt"].to_numpy()
  opt_sd      = df_full_pipeline["TSD_opt"].to_numpy()
  tutte       = df_full_pipeline["T_tutte"].to_numpy()
  tutte_sd    = df_full_pipeline["TSD_tutte"].to_numpy()
  project     = df_full_pipeline["T_project"].to_numpy()
  project_sd  = df_full_pipeline["TSD_project"].to_numpy()
  overhead    = df_full_pipeline["T_overhead"].to_numpy()
  overhead_sd = df_full_pipeline["TSD_overhead"].to_numpy()
  gen         = df_full_pipeline["T_gen"].to_numpy()
  gen_sd      = df_full_pipeline["TSD_gen"].to_numpy()
  dual        = df_full_pipeline["T_dual"].to_numpy()
  dual_sd     = df_full_pipeline["TSD_dual"].to_numpy()
  natoms      = df_full_pipeline["N"].to_numpy()


  parallel = opt + tutte + project
  parallel_sd = np.sqrt(opt_sd**2 + tutte_sd**2 + project_sd**2)
  total = parallel + overhead + gen + dual

  def plot_normalized_line(ax, x, y, y_sd, label, color, marker, linestyle, ms_scale=1.):
      ax.plot(x, 1e2* y/total, marker=marker, color=color, label=label, ls=linestyle, mfc=color, ms = ms_scale*rc["lines.markersize"]) #Normalized to total time, shown as percentage
      ax.fill_between(x, 1e2*(y - y_sd)/total, 1e2*(y + y_sd)/total, alpha=0.1, color='k')

  def plot_absolute_line(ax, x, y, y_sd, label, color, marker, linestyle, ms_scale=1.):
    ax.plot(x,  y/1e3, marker=marker, color=color, label=label, ls=linestyle, mfc=color, ms = ms_scale*rc["lines.markersize"]) #Absolute time
    ax.fill_between(x, (y - y_sd)/1e3, (y + y_sd)/1e3, alpha=0.1, color='k')

  def plot_logabsolute_line(ax, x, y, y_sd, label, color, marker, linestyle, ms_scale=1.):
    ax.set_yscale("log")
    ax.plot(x,  y/1e3, marker=marker, color=color, label=label, ls=linestyle, mfc=color, ms = ms_scale*rc["lines.markersize"]) #Set to log scale and plot absolute time
    ax.fill_between(x, (y - y_sd)/1e3 + ((y-y_sd)<0)*y_sd/1e3, (y + y_sd)/1e3, alpha=0.1, color='k')    

  if normalize:
      plot_fun = plot_normalized_line
  elif log:
      plot_fun = plot_logabsolute_line
  else:
      plot_fun = plot_absolute_line
      
  plot_fun(ax, natoms, gen, gen_sd, "Isomer-space graph generation", CD["Generate"], f'{Markers[0]}', ':', MarkerScales[0])
  plot_fun(ax, natoms, parallel, parallel_sd, "Lockstep-parallel geometry optimization", "k", f'{Markers[1]}', ':', MarkerScales[1])
  plot_fun(ax, natoms, overhead, overhead_sd, "Overhead", "blue", f'{Markers[2]}', ':', MarkerScales[2])
  plot_fun(ax, natoms, dual, dual_sd, "Lockstep-parallel dualization", CD["Dual"], f'{Markers[3]}', ':', MarkerScales[3])

  ax.set_ylabel(r"Runtime Fraction [$\%$]") if normalize else ax.set_ylabel(r"Time / Graph [$\mu$s]")
  
  ax.legend(loc='center') if normalize else ax.legend(bbox_to_anchor=(0.2, 0.6)) if log else ax.legend()

  ax.set_xlabel(r"Isomerspace $C_N$")
  if normalize:
    ax.set_ylim(0,100)
  #percentage formatting
  ax.yaxis.set_major_formatter(ticker.PercentFormatter()) if normalize else ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation()) if log else None
  normalized_str = "" if not normalize else "_normalized"
  log_str = "" if not log else "_log"  
  plt.savefig(f"{path}/figures/lockstep_pipeline{normalized_str}{log_str}.{save_format}", bbox_inches='tight')


def plot_speedup():
  MS = MarkerScales * rc["lines.markersize"]
  #print(f"Plotting single-GPU speedup benchmark from {relpath(fname_base,cwd)} and {relpath(fname_one_gpu_dual + "1.csv",cwd)} to {relpath(path,cwd)}/figures/speedup.{save_format}") 
  fig, ax = plt.subplots(figsize=(20,10), nrows=1, sharex=True)
  Nbase, Tbase, TSDbase = read_csv(fname_base)
  Ngpu, Tgpu, TSDgpu = read_csv(fname_one_gpu_dual + "1.csv")
  if Nbase is None or Ngpu is None: return

  speedup = Tbase / Tgpu
  speedup_sd = np.sqrt((TSDbase/Tbase)**2 + (TSDgpu/Tgpu)**2) * speedup

  ax.plot(Nbase, speedup, f'{Markers[0]}:', ms=MS[0], color="r", label="Speedup")
  ax.fill_between(Nbase, speedup - speedup_sd, speedup + speedup_sd, alpha=0.1, color='k')
  ax.set_ylabel(r"Speedup")
  ax.legend()
  ax.set_xlabel(r"Isomerspace $C_N$")
  plt.savefig(path + f"figures/speedup.{save_format}", bbox_inches='tight')


plot_kernel_cpu_benchmarks() #Can't use large markers for this plot, since data points are too close together
rc["lines.markersize"] = 8
plot_batch_size()
plot_baseline()
plot_weak_scaling()
plot_speedup()
plot_kernel_gpu_benchmarks()
set_fontsizes(40) #Single figure plots need larger fonts compared to multi-figure plots
rc["lines.markersize"] = 10
plot_pipeline(normalize=True)
plot_pipeline(normalize=False)
plot_lockstep_pipeline(normalize=True)
plot_lockstep_pipeline(normalize=False)
plot_lockstep_pipeline(normalize=False, log=True)



