import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, colorsys
from matplotlib import rcParams as rc
import os, sys,subprocess, platform
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset)
from os.path import relpath

rc["legend.markerscale"] = 2.0
rc["legend.framealpha"] = 0
rc["legend.labelspacing"] = 0.1
rc['figure.figsize'] = (20,10)
rc['axes.autolimit_mode'] = 'data'
rc['axes.xmargin'] = 0
rc['axes.ymargin'] = 0.10
rc['axes.titlesize'] = 30
rc['axes.labelsize'] = 24
rc['font.sans-serif'] = "Times New Roman"
rc['font.serif'] = "Times New Roman"
rc['xtick.labelsize'] = 20
rc['ytick.labelsize'] = 20
rc['axes.grid'] = True
rc['grid.linestyle'] = '-'
rc['grid.alpha'] = 0.2
rc['legend.fontsize'] = 20
rc['legend.loc'] = 'upper left'
rc["figure.autolayout"] = True
rc["savefig.dpi"] = 300
rc["text.usetex"] = True
rc["font.family"] = "Times New Roman"
rc.update({
  "text.usetex": True,
  "font.family": "Times New Roman"
})

if(len(sys.argv)>1):
    benchname = sys.argv[1]
else:
    benchname = platform.node()

# Benchmark result filenames
cwd = os.getcwd()
path = f'/{cwd}/output/{benchname}/'
buildpath = f'/{cwd}/build/'
fname_base = f'{path}base.csv'
fname_one_gpu_v0 = f'{path}one_gpu_v0.csv'
fname_one_gpu_v1 = f'{path}one_gpu_v1.csv'
fname_multi_gpu_v0 = f'{path}multi_gpu_v0.csv'
fname_multi_gpu_v1 = f'{path}multi_gpu_v1.csv'
fname_multi_gpu_weak = f'{path}multi_gpu_weak.csv'
fname_single_gpu_bs = f'{path}single_gpu_bs.csv'
fname_single_gpu_bs = f'{path}single_gpu_bs.csv'
fname_base_pipeline = f'{path}base_pipeline.csv'
fname_full_pipeline = f'{path}full_pipeline.csv'

os.makedirs(f"{path}/figures",exist_ok=True)

colors = ["#1f77b4", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
CD = { "Baseline" : colors[0], "GPU_V0" : colors[1], "GPU_V1" : colors[2], "2 GPU_V0" :  colors[3] ,"2 GPU_V1" : colors[4]}

def adjust_brightness(color, amount):
    """Adjust the brightness of a color by a given amount (-1 to 1)."""
    # Convert the color to the RGB color space
    r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    # Convert the color to the HLS color space
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    # Modify the lightness value
    l = max(0, min(1, l + amount))
    # Convert the color back to RGB and return it
    r, g, b = tuple(round(c * 255) for c in colorsys.hls_to_rgb(h, l, s))
    return f"#{r:02x}{g:02x}{b:02x}"
# Modify the brightness of the colors
colors = [adjust_brightness(color, 0.2) for color in colors]

CD = { "Baseline" : 'r', "GPU_V0" : colors[2], "GPU_V1" : colors[3], "2 GPU_V0" :  colors[4] ,"2 GPU_V1" : colors[0], "Dual" : f'#7570b3', "Generate" : f'#d95f02', "Projection" : f'#e7298a', "Tutte" : f'#66a61e', "Opt" : f'#8931EF' }

KName0 = r"SYCL Kernel 0"
KName1 = r"SYCL Kernel 1"

## Batch size

def plot_batch_size():
  print(f"Plotting batch size benchmark from {relpath(fname_single_gpu_bs,cwd)} to {relpath(path,cwd)}/figures/batch_size_benchmark.pdf")
  df_single_gpu_bs = pd.read_csv(fname_single_gpu_bs)
  Nrows = df_single_gpu_bs.shape[0]
  fig,ax = plt.subplots(figsize=(15,10))
  def add_line(ax, BS, T, SD, label, color, marker, linestyle):
      ax.plot(BS, T, marker=marker, color=color, label=label, linestyle=linestyle)
      ax.fill_between(BS, T - SD, T + SD, alpha=0.1, color='k')
  
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_ylabel("Time / Graph [ns]")
  ax.set_xlabel("Batch Size")
  add_line(ax, df_single_gpu_bs["BS"].to_numpy(), df_single_gpu_bs["T"].to_numpy(), df_single_gpu_bs["TSD"].to_numpy(), "Lockstep Parallel Dualization", CD["GPU_V1"], 'o', ':')
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
  add_line(axins, df_single_gpu_bs["BS"].to_numpy()[-7:], df_single_gpu_bs["T"].to_numpy()[-7:], df_single_gpu_bs["TSD"].to_numpy()[-7:], "Lockstep Parallel Dualization", CD["GPU_V1"], 'o', ':')
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

  plt.savefig(f'{path}figures/batch_size_benchmark.pdf', bbox_inches='tight')


## Baseline Sequential Dualization
def plot_baseline():
  print(f"Plotting baseline benchmark from {relpath(fname_base,cwd)} to {relpath(path,cwd)}/figures/baseline.pdf")    
  df_base = pd.read_csv(fname_base)

  fig, ax = plt.subplots(figsize=(15,15), nrows=2, sharex=True, dpi=200)
  ax[0].plot(df_base["N"].to_numpy(), df_base["T"].to_numpy()/1e3, 'o:', color=CD["Baseline"], label="Baseline Sequential Dualisation")
  ax[0].fill_between(df_base["N"].to_numpy(), (df_base["T"].to_numpy() - df_base["TSD"].to_numpy())/1e3, (df_base["T"].to_numpy() + df_base["TSD"].to_numpy())/1e3, alpha=0.1, color='k')
  ax[0].set_ylabel(r"Time / Graph [$\mu$s]")
  ax[0].legend()

  ax[1].plot(df_base["N"].to_numpy(), df_base["T"].to_numpy() / df_base["N"].to_numpy(), 'o:', color=CD["Baseline"], label="Baseline Sequential Dualisation")
  ax[1].fill_between(df_base["N"].to_numpy(), (df_base["T"].to_numpy() - df_base["TSD"].to_numpy()*2) / df_base["N"].to_numpy(), (df_base["T"].to_numpy()+df_base["TSD"].to_numpy()*2) / df_base["N"].to_numpy(), color='k', alpha=0.1, label=r"2$\sigma$")
  ax[1].set_ylabel(r"Time / Vertex [ns]")
  ax[1].set_xlabel(r"Cubic Graph Size [\# Vertices]")
  ax[1].legend()
  plt.savefig(path + "figures/baseline.pdf", bbox_inches='tight')

##

def plot_weak_scaling():
  print(f"Plotting scaling benchmark to {relpath(path,cwd)}/figures/dual_gpu_scaling.pdf")
  df1 = pd.read_csv(fname_one_gpu_v1)
  df3 = pd.read_csv(fname_multi_gpu_v1)
  df2 = pd.read_csv(fname_multi_gpu_weak)
  def std_div(a,b, a_std, b_std):
      return a/b * np.sqrt((a_std/a)**2 + (b_std/b)**2)
  fig, ax     = plt.subplots(figsize=(15, 15), nrows=2, sharex=True)
  ax[0].plot(df1["N"].to_numpy(), df1["T"].to_numpy(), 'o:',  color=CD["GPU_V1"], label=f"1 GPU, $B_s = 2^{{{20}}}$")
  ax[0].plot(df3["N"].to_numpy(), df3["T"].to_numpy(), 'o:',  color=CD["2 GPU_V1"], label=f"2 GPUs $B_s = 2^{{{21}}}$")
  ax[0].plot(df2["N"].to_numpy(), df2["T"].to_numpy(), 'x--',  color=CD["2 GPU_V1"], label=f"2 GPUs $B_s = 2^{{{20}}}$")
  ax[0].fill_between(df1["N"].to_numpy(), df1["T"].to_numpy() - df1["TSD"].to_numpy()*1, df1["T"].to_numpy() + df1["TSD"].to_numpy()*1, alpha=0.1, color='k', label=r"1$\sigma$")
  ax[0].fill_between(df3["N"].to_numpy(), df3["T"].to_numpy() - df3["TSD"].to_numpy()*1, df3["T"].to_numpy() + df3["TSD"].to_numpy()*1, alpha=0.1, color='k')
  ax[0].fill_between(df2["N"].to_numpy(), df2["T"].to_numpy() - df2["TSD"].to_numpy()*1, df2["T"].to_numpy() + df2["TSD"].to_numpy()*1, alpha=0.1, color='k')
  ax[0].set_ylabel("Time / Graph [ns]")
  ax[0].legend(loc='upper left')


  #Plot speedup
  ax[1].plot(df1["N"].to_numpy(), df1["T"].to_numpy()/df3["T"].to_numpy(), 'o:',  color=CD["2 GPU_V1"], label=f"2 GPUs $B_s = 2^{{{21}}}$")
  ax[1].plot(df1["N"].to_numpy(), df1["T"].to_numpy()/df2["T"].to_numpy(), 'x--',  color=CD["2 GPU_V1"], label=f"2 GPUs $B_s = 2^{{{20}}}$")
  std_1 = std_div(df1["T"].to_numpy(), df3["T"].to_numpy(), df1["TSD"].to_numpy(), df3["TSD"].to_numpy())
  std_2 = std_div(df1["T"].to_numpy(), df2["T"].to_numpy(), df1["TSD"].to_numpy(), df2["TSD"].to_numpy())
  ax[1].fill_between(df1["N"].to_numpy(), df1["T"].to_numpy()/df3["T"].to_numpy() - std_1, df1["T"].to_numpy()/df3["T"].to_numpy() + std_1, alpha=0.1, color='k', label=r"1$\sigma$")
  ax[1].fill_between(df1["N"].to_numpy(), df1["T"].to_numpy()/df2["T"].to_numpy() - std_2, df1["T"].to_numpy()/df2["T"].to_numpy() + std_2, alpha=0.1, color='k')
  ax[1].hlines(2, 20, 200, linestyles='dashed', color='k', label=r"Perfect Scaling")
  ax[1].set_ylabel("Speedup")
  ax[1].set_xlabel(r"Cubic Graph Size [\# Vertices]")
  ax[1].set_ylim(0.95,2*1.05)
  ax[1].legend(loc='lower right', ncol=2)
  plt.savefig(path + "figures/dual_gpu_scaling.pdf", bbox_inches='tight')

def plot_pipeline(normalize=False):
  print(f"Plotting pipeline benchmark from {relpath(fname_base_pipeline,cwd)} to {relpath(path,cwd)}/figures/pipeline.pdf")
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

  def plot_normalized_line(ax, x, y, y_sd, label, color, marker, linestyle, mfc_bool=True):
      if mfc_bool:
          ax.plot(x, 1e2* y/total, marker=marker, color=color, label=label, linestyle=linestyle, mfc=color) #Normalized to total time, shown as percentage
      else:
          ax.plot(x, 1e2* y/total, marker=marker, color=color, label=label, linestyle=linestyle, mfc="None") #Normalized to total time, shown as percentage
      ax.fill_between(x, 1e2*(y - y_sd)/total, 1e2*(y + y_sd)/total, alpha=0.1, color='k')

  def plot_absolute_line(ax, x, y, y_sd, label, color, marker, linestyle, mfc_bool=True):
    if mfc_bool:
        ax.plot(x,  y/1e3, marker=marker, color=color, label=label, linestyle=linestyle, mfc=color) #Normalized to total time, shown as percentage
    else:
        ax.plot(x,  y/1e3, marker=marker, color=color, label=label, linestyle=linestyle, mfc="None") #Normalized to total time, shown as percentage
    ax.fill_between(x, (y - y_sd)/1e3, (y + y_sd)/1e3, alpha=0.1, color='k')
  if normalize:
    plot_normalized_line(ax, natoms, gen, gen_sd, "Isomer-space graph generation", CD["Generate"], 'o', ':', False)
    plot_normalized_line(ax, natoms, parallel, parallel_sd, "Lockstep-parallel geometry optimization", "k", '*', ':')
    plot_normalized_line(ax, natoms, overhead, overhead_sd, "Overhead", "blue", 'o', ':')
    plot_normalized_line(ax, natoms, dual, dual_sd, "Baseline Sequential Dualization", CD["Dual"], '*', ':')
  else:
    plot_absolute_line(ax, natoms, parallel, parallel_sd, "Lockstep-parallel geometry optimization", "k", '*', ':')
    plot_absolute_line(ax, natoms, gen, gen_sd, "Isomer-space graph generation", CD["Generate"], 'o', ':', False)
    plot_absolute_line(ax, natoms, overhead, overhead_sd, "Overhead", "blue", 'o', ':')
    plot_absolute_line(ax, natoms, dual, dual_sd, "Lockstep-parallel dualization", CD["Dual"], '*', ':')
      

  if normalize:
    plot_normalized_line(ax, natoms, gen, gen_sd, "Isomer-space graph generation", CD["Generate"], 'o', ':', False)
    plot_normalized_line(ax, natoms, parallel, parallel_sd, "Lockstep-parallel geometry optimization", "k", '*', ':')
    plot_normalized_line(ax, natoms, overhead, overhead_sd, "Overhead", "blue", 'o', ':')
    plot_normalized_line(ax, natoms, dual, dual_sd, "Baseline Sequential Dualization", CD["Dual"], '*', ':')
  else:
    plot_absolute_line(ax, natoms, parallel, parallel_sd, "Lockstep-parallel geometry optimization", "k", '*', ':')
    plot_absolute_line(ax, natoms, gen, gen_sd, "Isomer-space graph generation", CD["Generate"], 'o', ':', False)
    plot_absolute_line(ax, natoms, overhead, overhead_sd, "Overhead", "blue", 'o', ':')
    plot_absolute_line(ax, natoms, dual, dual_sd, "Lockstep-parallel dualization", CD["Dual"], '*', ':')

  ax.set_ylabel(r"Runtime Fraction [$\%$]") if normalize else ax.set_ylabel(r"Time / Graph [$\mu$s]")
  ax.legend()
  ax.set_xlabel(r"Isomerspace $C_N$")
  ax.set_ylim(0,100)
  #percentage formatting
  ax.yaxis.set_major_formatter(ticker.PercentFormatter())
  plt.savefig(path + "figures/pipeline.pdf", bbox_inches='tight')

def plot_lockstep_pipeline(normalize=False):
  print(f"Plotting lockstep pipeline benchmark from {relpath(fname_full_pipeline,cwd)} to {relpath(path,cwd)}/figures/lockstep_pipeline.pdf")
  df_full_pipeline = pd.read_csv(fname_full_pipeline)

  fig, ax = plt.subplots(figsize=(20,10), nrows=1, sharex=True)

  opt = df_full_pipeline["T_opt"].to_numpy()
  opt_sd = df_full_pipeline["TSD_opt"].to_numpy()
  tutte = df_full_pipeline["T_tutte"].to_numpy()
  tutte_sd = df_full_pipeline["TSD_tutte"].to_numpy()
  project = df_full_pipeline["T_project"].to_numpy()
  project_sd = df_full_pipeline["TSD_project"].to_numpy()
  overhead = df_full_pipeline["T_overhead"].to_numpy()
  overhead_sd = df_full_pipeline["TSD_overhead"].to_numpy()
  gen = df_full_pipeline["T_gen"].to_numpy()
  gen_sd = df_full_pipeline["TSD_gen"].to_numpy()
  dual = df_full_pipeline["T_dual"].to_numpy()
  dual_sd = df_full_pipeline["TSD_dual"].to_numpy()
  natoms = df_full_pipeline["N"].to_numpy()


  parallel = opt + tutte + project
  parallel_sd = np.sqrt(opt_sd**2 + tutte_sd**2 + project_sd**2)
  total = parallel + overhead + gen + dual

  def plot_normalized_line(ax, x, y, y_sd, label, color, marker, linestyle, mfc_bool=True):
      if mfc_bool:
          ax.plot(x, 1e2* y/total, marker=marker, color=color, label=label, linestyle=linestyle, mfc=color) #Normalized to total time, shown as percentage
      else:
          ax.plot(x, 1e2* y/total, marker=marker, color=color, label=label, linestyle=linestyle, mfc="None") #Normalized to total time, shown as percentage
      ax.fill_between(x, 1e2*(y - y_sd)/total, 1e2*(y + y_sd)/total, alpha=0.1, color='k')

  def plot_absolute_line(ax, x, y, y_sd, label, color, marker, linestyle, mfc_bool=True):
    if mfc_bool:
        ax.plot(x,  y/1e3, marker=marker, color=color, label=label, linestyle=linestyle, mfc=color) #Normalized to total time, shown as percentage
    else:
        ax.plot(x,  y/1e3, marker=marker, color=color, label=label, linestyle=linestyle, mfc="None") #Normalized to total time, shown as percentage
    ax.fill_between(x, (y - y_sd)/1e3, (y + y_sd)/1e3, alpha=0.1, color='k')
  if normalize:
    plot_normalized_line(ax, natoms, gen, gen_sd, "Isomer-space graph generation", CD["Generate"], 'o', ':', False)
    plot_normalized_line(ax, natoms, parallel, parallel_sd, "Lockstep-parallel geometry optimization", "k", '*', ':')
    plot_normalized_line(ax, natoms, overhead, overhead_sd, "Overhead", "blue", 'o', ':')
    plot_normalized_line(ax, natoms, dual, dual_sd, "Baseline Sequential Dualization", CD["Dual"], '*', ':')
  else:
    plot_absolute_line(ax, natoms, parallel, parallel_sd, "Lockstep-parallel geometry optimization", "k", '*', ':')
    plot_absolute_line(ax, natoms, gen, gen_sd, "Isomer-space graph generation", CD["Generate"], 'o', ':', False)
    plot_absolute_line(ax, natoms, overhead, overhead_sd, "Overhead", "blue", 'o', ':')
    plot_absolute_line(ax, natoms, dual, dual_sd, "Lockstep-parallel dualization", CD["Dual"], '*', ':')

  ax.set_ylabel(r"Runtime Fraction [$\%$]") if normalize else ax.set_ylabel(r"Time / Graph [$\mu$s]")
  ax.legend()
  ax.set_xlabel(r"Isomerspace $C_N$")
  if normalize:
    ax.set_ylim(0,100)
  #percentage formatting
  ax.yaxis.set_major_formatter(ticker.PercentFormatter()) if normalize else ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
  normalized_str = "" if not normalize else "_normalized"
  plt.savefig(path + "figures/lockstep_pipeline" + normalized_str + ".pdf", bbox_inches='tight')


def plot_speedup():
  print(f"Plotting single-GPU speedup benchmark from {relpath(fname_base,cwd)} and {relpath(fname_one_gpu_v1,cwd)} to {relpath(path,cwd)}/figures/speedup.pdf") 
  fig, ax = plt.subplots(figsize=(20,10), nrows=1, sharex=True)
  df_baseline = pd.read_csv(fname_base)
  df_dual_lockstep = pd.read_csv(fname_one_gpu_v1)

  parallel = df_dual_lockstep["T"].to_numpy()
  parallel_sd = df_dual_lockstep["TSD"].to_numpy()
  sequential = df_baseline["T"].to_numpy()
  sequential_sd = df_baseline["TSD"].to_numpy()

  natoms = df_dual_lockstep["N"].to_numpy()

  speedup = sequential / parallel
  speedup_sd = np.sqrt((sequential_sd/sequential)**2 + (parallel_sd/parallel)**2) * speedup

  ax.plot(natoms, speedup, 'o:', color="r", label="Speedup")
  ax.fill_between(natoms, speedup - speedup_sd, speedup + speedup_sd, alpha=0.1, color='k')
  ax.set_ylabel(r"Speedup")
  ax.legend()
  ax.set_xlabel(r"Isomerspace $C_N$")
  plt.savefig(path + "figures/speedup.pdf", bbox_inches='tight')

plot_batch_size()
plot_baseline()
plot_weak_scaling()
plot_pipeline(normalize=True)
plot_pipeline(normalize=False)
plot_lockstep_pipeline(normalize=True)
plot_lockstep_pipeline(normalize=False)
plot_speedup()



