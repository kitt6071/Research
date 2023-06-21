
import numpy as np
import scipy.ndimage as ndi
import scipy.stats as stats
import matplotlib.pyplot as plt
import xarray as xr

def label_precipitation_clusters(da, threshold):
    structure = np.ones((3, 3))  
    labeled_array = np.empty(da.shape, dtype=int)
    num_clusters_list = []
    for t in range(da.sizes['time']):
        mask = da[t] > threshold
        labeled_array_t, num_clusters = ndi.label(mask, structure)
        labeled_array[t] = labeled_array_t
        num_clusters_list.append(num_clusters)
        
    return xr.DataArray(labeled_array, coords=da.coords), num_clusters_list



def compute_cutoff_moment_ratio(cluster_array):
    mean_value = cluster_array.mean().values
    second_moment = (cluster_array ** 2).mean().values

    cutoff = second_moment / mean_value
    return cutoff

def compute_cutoff_linear_regression(cluster_array):
    hist, bins = np.histogram(cluster_array.values, bins=100, density=True)
    
    log_hist = np.log(hist) 
    valid = log_hist != -np.inf
    valid_bins = bins[:-1][valid]  
    valid_log_hist = log_hist[valid]

    slope, intercept, _, _, _ = stats.linregress(valid_bins[-20:], valid_log_hist[-20:])
    cutoff = -1 / slope

    return cutoff, valid_bins, valid_log_hist, slope, intercept

def calculate_cutoffs_over_time(da, structure, start_day, end_day, increment):
    cutoffs_moment_ratio = [] 
    cutoffs_linear_regression = []

    for day in range(start_day, end_day, increment):
        da_day = da.sel(time=slice(day, day+increment-1))

        if da_day.size == 0:
            continue

        if da_day.isnull().any():
            da_day = da_day.fillna(0)

        cutoff_moment_ratio = compute_cutoff_moment_ratio(da_day)
        cutoff_linear_regression, _, _, _, _ = compute_cutoff_linear_regression(da_day)
        cutoffs_moment_ratio.append(cutoff_moment_ratio)
        cutoffs_linear_regression.append(cutoff_linear_regression)

    return cutoffs_moment_ratio, cutoffs_linear_regression



ds = xr.open_dataset('Ts300_warm_precip.nc', decode_times=False)
da = ds["prec_mp"]  
if da.isnull().any():
    da = da.fillna(0)
structure = np.ones((3)) 
def calculate_cutoffs(data, percentile=95):
    return np.percentile(data, percentile)

threshold = 1

precip_clusters, num_clusters = label_precipitation_clusters(da, threshold)

cutoff_moment_ratio = compute_cutoff_moment_ratio(precip_clusters)
cutoff_linear_regression, bins, log_hist, slope, intercept = compute_cutoff_linear_regression(precip_clusters)

print(f"Cutoff (Moment Ratio Method): {cutoff_moment_ratio}")
print(f"Cutoff (Linear Regression Method): {cutoff_linear_regression}")

fig, ax = plt.subplots()

ax.plot(bins, np.exp(log_hist), label="Data")

tail_x = np.linspace(bins[-20], bins[-1], 100)
tail_y = slope * tail_x + intercept
ax.plot(tail_x, np.exp(tail_y), label="Linear fit on tail")

ax.set_yscale('log')
ax.set_xlabel("Cluster Size or Power")
ax.set_ylabel("Probability Density")
ax.legend()

plt.show()
cutoffs_moment_ratio, cutoffs_linear_regression = calculate_cutoffs_over_time(da, structure, start_day=61, end_day=120, increment=1)

plt.figure(figsize=(10, 5))
start_day = 61
end_day = 120
days_range = range(start_day, end_day)  
plt.plot(days_range, cutoffs_moment_ratio, label='Moment Ratio Method')
plt.plot(days_range, cutoffs_linear_regression, label='Linear Regression Method')
plt.xlabel('Day')
plt.ylabel('Cutoff')
plt.title('Cutoffs over time')
plt.legend()
plt.show()

mask = da > threshold
print(mask.shape)

structure = np.ones((3, 3, 3))  
print(structure.shape)

labeled_array, num_clusters = ndi.label(mask, structure)

cluster_sizes = np.bincount(labeled_array.ravel())

largest_clusters_indices = cluster_sizes.argsort()[-10:]  

largest_clusters = np.isin(labeled_array, largest_clusters_indices)

plt.figure(figsize=(10, 5))
plt.imshow(largest_clusters[0], cmap='viridis')  
plt.title('Largest clusters')
plt.colorbar(label='Cluster label')
plt.show()

def largest_cluster_size_each_timestep(cluster_array):

    cluster_sizes = [np.bincount(timestep.ravel()) for timestep in cluster_array]
    largest_sizes = [sizes.max() for sizes in cluster_sizes]

    return largest_sizes

largest_sizes = largest_cluster_size_each_timestep(labeled_array)

plt.figure(figsize=(10, 5))
plt.plot(largest_sizes)
plt.xlabel('Time step')
plt.ylabel('Size of largest cluster')
plt.title('Size of largest cluster over time')
plt.show()

def plot_cluster_size_histogram(cluster_array, timestep):

    sizes = np.bincount(cluster_array[timestep].ravel())
    sizes = sizes[1:]
    plt.figure(figsize=(10, 5))
    plt.hist(sizes, bins=100)
    plt.xlabel('Cluster size')
    plt.ylabel('Count')
    plt.title(f'Cluster size distribution at time step {timestep}')
    plt.show()

plot_cluster_size_histogram(labeled_array, timestep=30)  
