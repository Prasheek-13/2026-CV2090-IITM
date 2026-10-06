import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# parameters
mu = 6
sigma = 1
N = 30
n_iter = 100
np.random.seed(42)  

# storage
x_samples = []
x_max_samples = []
x_min_samples = []

# save options
plotMaximaOrMinima = "both"  # "maxima" or "minima" or "both"
doSavePlot = True

# generate samples
for i in range(n_iter):
    x = np.random.normal(mu, sigma, N)
    x_samples.extend(x)              # keep all N samples
    x_max_samples.append(np.max(x))  # store maximum of each batch
    x_min_samples.append(np.min(x))  # store minimum of each batch


# convert to arrays
x_samples = np.array(x_samples)
x_max_samples = np.array(x_max_samples)
x_min_samples = np.array(x_min_samples)

# helper: ECDF
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

# figure with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# --- Subplot 1: scatter of all samples + maxima ---
axs[0, 0].scatter(range(len(x_samples)), x_samples, 
                  color="steelblue", s=10, alpha=0.6, label="All samples")

# find index of max and min in each bucket
max_indices = [i*N + np.argmax(x_samples[i*N:(i+1)*N]) for i in range(n_iter)]
min_indices = [i*N + np.argmin(x_samples[i*N:(i+1)*N]) for i in range(n_iter)]

if plotMaximaOrMinima == "maxima":
    axs[0, 0].scatter(max_indices, x_max_samples, 
                    color="darkorange", s=40, label="Maxima (per bucket)")
elif plotMaximaOrMinima == "minima":
    axs[0, 0].scatter(min_indices, x_min_samples, 
                  color="crimson", s=40, label="Minima (per bucket)")
elif plotMaximaOrMinima == "both":
    axs[0, 0].scatter(max_indices, x_max_samples, 
                    color="darkorange", s=40, label="Maxima (per bucket)")
    axs[0, 0].scatter(min_indices, x_min_samples, 
                  color="crimson", s=40, label="Minima (per bucket)")

axs[0, 0].set_xlabel("Index", fontsize=18)
axs[0, 0].set_ylabel("X", fontsize=18)
axs[0, 0].grid(alpha=0.5)
axs[0, 0].tick_params(axis="both", labelsize=18)  
# axs[0, 0].set_title("(1) All samples + maxima")
# axs[0, 0].legend(fontsize=18, loc="lower right", title=None)  



# --- Subplot 2: zoomed scatter (first 4 buckets = 120 points) ---
axs[0, 1].scatter(range(4*N), x_samples[:4*N], 
                  color="steelblue", s=20, alpha=0.6, label="Samples")

# overlay maxima and minima on top of existing sample points
max_indices_zoom = [i*N + np.argmax(x_samples[i*N:(i+1)*N]) for i in range(4)]
min_indices_zoom = [i*N + np.argmin(x_samples[i*N:(i+1)*N]) for i in range(4)]

if plotMaximaOrMinima == "maxima":
    axs[0, 1].scatter(max_indices_zoom, x_max_samples[:4], 
                  color="darkorange", s=60, label="Maxima", zorder=3)
elif plotMaximaOrMinima == "minima":
    axs[0, 1].scatter(min_indices_zoom, x_min_samples[:4], 
                  color="crimson", s=60, label="Minima", zorder=3)
elif plotMaximaOrMinima == "both":
    axs[0, 1].scatter(max_indices_zoom, x_max_samples[:4], 
                  color="darkorange", s=60, label="Maxima", zorder=3)
    axs[0, 1].scatter(min_indices_zoom, x_min_samples[:4], 
                  color="crimson", s=60, label="Minima", zorder=3)    

# add grayish backgrounds for buckets 1 & 3
axs[0, 1].axvspan(0, N, facecolor="lightgray", alpha=0.3)
axs[0, 1].axvspan(2*N, 3*N, facecolor="lightgray", alpha=0.3)

axs[0, 1].set_xlabel("Index", fontsize=18)
axs[0, 1].set_ylabel("X", fontsize=18)
axs[0, 1].grid(alpha=0.5)
axs[0, 1].tick_params(axis="both", labelsize=18)  
# axs[0, 1].set_title("(2) Zoom: first 4 buckets")
# axs[0, 1].legend(loc="lower right")


# --- Subplot 3: histogram with twin y-axis ---
ax3 = axs[1, 0]
ax3.hist(x_samples, bins=20, alpha=0.6, color="steelblue", label="All samples")
ax3.set_xlabel("X", fontsize=18)
ax3.set_ylabel("Frequency (all samples)", color="steelblue", fontsize=18)
# ax3.set_title("(3) Histograms with twin y-axis")
axs[1, 0].grid(alpha=0.5)
axs[1, 0].tick_params(axis="both", labelsize=18)  

ax3b = ax3.twinx()
if plotMaximaOrMinima == "maxima":
    ax3b.hist(x_max_samples, bins=20, alpha=0.6, color="darkorange", label="Maxima")
    ax3b.set_ylabel("Frequency (maxima)", color="darkorange", fontsize=18)
elif plotMaximaOrMinima == "minima":
    ax3b.hist(x_min_samples, bins=20, alpha=0.6, color="crimson", label="Minima")
    ax3b.set_ylabel("Frequency (minima)", color="crimson", fontsize=18)
elif plotMaximaOrMinima == "both":
    ax3b.hist(x_max_samples, bins=20, alpha=0.6, color="darkorange", label="Maxima")
    ax3b.hist(x_min_samples, bins=20, alpha=0.6, color="crimson", label="Minima")
    ax3b.set_ylabel("Frequency (maxima & minima)", color="black", fontsize=18)


ax3b.tick_params(axis="both", labelsize=18)  

# --- Subplot 4: ECDFs ---
x_all, y_all = ecdf(x_samples)
x_max, y_max = ecdf(x_max_samples)
x_min, y_min = ecdf(x_min_samples)

axs[1, 1].step(x_all, y_all, where="post", color="steelblue", label="ECDF (all)")
if plotMaximaOrMinima == "maxima":
    axs[1, 1].step(x_max, y_max, where="post", color="darkorange", label="ECDF (maxima)")
elif plotMaximaOrMinima == "minima":
    axs[1, 1].step(x_min, y_min, where="post", color="crimson", label="ECDF (minima)")
elif plotMaximaOrMinima == "both":
    axs[1, 1].step(x_max, y_max, where="post", color="darkorange", label="ECDF (maxima)")
    axs[1, 1].step(x_min, y_min, where="post", color="crimson", label="ECDF (minima)")

axs[1, 1].set_xlabel("X", fontsize=18)
axs[1, 1].set_ylabel("ECDF", fontsize=18)
axs[1, 1].grid(alpha=0.5)
axs[1, 1].tick_params(axis="both", labelsize=18)  
# axs[1, 1].set_title("(4) Empirical CDFs")
# axs[1, 1].legend(fontsize=16, loc="upper left")

plt.tight_layout()

if (doSavePlot == True):
    plt.savefig(fr"gev_simulation_{plotMaximaOrMinima}.png", dpi=120, bbox_inches='tight')

plt.show()              # only pop a window when there's no saved file to look at