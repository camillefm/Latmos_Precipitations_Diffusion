import os
import numpy as np
import matplotlib.pyplot as plt

def plot_rain_rate_distribution(
    folder_paths,
    output_path='/net/nfs/ssd3/cfrancoismartin/Projects/datasets/results/rain_rate_distribution.png',
    custom_bins=None,
    log_scale=True,
    show=False,
    transform_fn=None,
    title="Distribution of 'rain_rate' Values"
):
    """
    Plots and saves the histogram of 'rain_rate' values across .npy files in the given folders,
    using optional custom bins and transformation.

    Parameters:
        folder_paths (list of str): List of folder paths to scan for .npy files.
        output_path (str): File path to save the histogram image.
        custom_bins (list of tuple): List of (min, max) tuples defining custom bins.
        transform_fn (callable): Optional function to apply to the 'rain_rate' values before plotting.
        log_scale (bool): If True, apply log scale to y-axis.
        show (bool): If True, display the plot interactively.
        title (str): Title for the plot.

    Returns:
        all_values (np.ndarray): Concatenated array of all valid (transformed) rain_rate values.
    """
    all_values = []

    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        data = np.load(file_path, allow_pickle=True).item()

                        if 'rain_rate' not in data:
                            continue

                        r = np.array(data['rain_rate'], dtype=np.float32)
                        r = r[~np.isnan(r)].flatten()

                        if transform_fn:
                            r = transform_fn(r)

                        all_values.append(r)

                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    if not all_values:
        print("No valid 'rain_rate' data found.")
        return np.array([])

    all_values = np.concatenate(all_values)

    # Plot with custom bins
    plt.figure(figsize=(10, 6))
    if custom_bins:
        bin_edges = sorted(set([b for tup in custom_bins for b in tup]))
        bin_labels = [f"[{a}, {b})" for (a, b) in custom_bins]
        hist_values = np.histogram(all_values, bins=bin_edges)[0]
        plt.bar(bin_labels, hist_values, color='skyblue', edgecolor='black')
        plt.xticks(rotation=45)
        plt.xlabel("Rain rate bins (custom)")
    else:
        plt.hist(all_values, bins=300, color='skyblue', edgecolor='black')
        plt.xlabel("Rain rate")
        # plt.xlim(-1, 1)

    plt.title(title)
    plt.ylabel("Frequency (log scale)" if log_scale else "Frequency")

    if log_scale:
        plt.yscale('log')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Histogram saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return all_values

import os
import numpy as np

def count_rain_rate_shapes(
    folder_paths,
    transform_fn=None,
    verbose=False
):
    """
    Counts the number of rain_rate arrays with shapes 128x128 and 160x160 across .npy files in the given folders.

    Parameters:
        folder_paths (list of str): List of folder paths to scan for .npy files.
        transform_fn (callable): Optional function to apply to the 'rain_rate' values before inspecting shape.
        verbose (bool): If True, print shape details for each file.

    Returns:
        dict: A dictionary with shape counts, e.g. {(128, 128): 42, (160, 160): 17}
    """
    shape_counts = {}

    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        data = np.load(file_path, allow_pickle=True).item()

                        if 'rain_rate' not in data:
                            print(f"Skipping {file_path}: 'rain_rate' not found.", flush =True)
                            continue

                        r = np.array(data['rain_rate'], dtype=np.float32)
                        r = r[~np.isnan(r)]

                        if transform_fn:
                            r = transform_fn(r)
                        print(r.shape, flush=True)
                        shape = r.shape[-1]
                        

                        if verbose:
                            print(f"{file_path}: shape {shape}")

                        shape_counts[shape] = shape_counts.get(shape, 0) + 1

                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    count_128 = shape_counts.get((16384), 0)
    count_160 = shape_counts.get((25600), 0)

    print(f"Count of (128, 128): {count_128}")
    print(f"Count of (160, 160): {count_160}")

    return {
        (128, 128): count_128,
        (160, 160): count_160,
        "all_shapes": shape_counts
    }


import os
import numpy as np

def get_mean_std_rain_rate(folder_paths, exclude_zeros=False):
    """
    Computes the mean and standard deviation of 'rain_rate' values across .npy files in the given folders.

    Parameters:
        folder_paths (list of str): List of folder paths to scan for .npy files.
        exclude_zeros (bool): If True, ignore zero values when computing statistics.

    Returns:
        mean (float): Mean of all valid 'rain_rate' values.
        std (float): Standard deviation of all valid 'rain_rate' values.
    """
    all_values = []

    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        data = np.load(file_path, allow_pickle=True).item()

                        if 'rain_rate' not in data:
                            continue

                        r = np.array(data['rain_rate'], dtype=np.float32)
                        r = r[~np.isnan(r)].flatten()

                        if exclude_zeros:
                            r = r[r >= 0.1]

                        if r.size > 0:
                            all_values.append(r)

                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    if not all_values:
        print("No valid 'rain_rate' data found.")
        return None, None

    all_values = np.concatenate(all_values)
    mean = np.mean(all_values)
    std = np.std(all_values)

    return mean, std

def get_min_max_rain_rate(folder_paths):
    """
    Computes the minimum and maximum 'rain_rate' values across .npy files in the given folders.

    Parameters:
        folder_paths (list of str): List of folder paths to scan for .npy files.

    Returns:
        min_value (float): Minimum 'rain_rate' value found.
        max_value (float): Maximum 'rain_rate' value found.
    """
    min_value = float('inf')
    max_value = float('-inf')

    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        data = np.load(file_path, allow_pickle=True).item()

                        if 'rain_rate' not in data:
                            continue

                        r = np.array(data['rain_rate'], dtype=np.float32)
                        r = r[~np.isnan(r)].flatten()

                        if r.size > 0:
                            min_value = min(min_value, r.min())
                            max_value = max(max_value, r.max())

                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    return min_value, max_value

def number_elements_in_image(image, threshold=0.1):
    """
    counts the number of element above a certain threshold in the image"""

    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    
    count = np.sum(image > threshold)
    return count

def plot_distribution_of_nb_elements(folder_paths, threshold=0.1, output_path=None, show=False, bins=50, title=None):
    """
    Computes the distribution of the number of elements above a certain threshold in 'rain_rate' images,
    plots and saves the histogram.

    Parameters:
        folder_paths (list of str): List of folder paths to scan for .npy files.
        threshold (float): Threshold value to count elements.
        output_path (str): Path to save the histogram image. If None, does not save.
        show (bool): If True, display the plot interactively.
        bins (int): Number of bins for the histogram.
        title (str): Title for the plot.

    Returns:
        distribution (dict): Dictionary with file names as keys and counts as values.
    """
    distribution = {}

    for folder_path in folder_paths:
        npy_files = [f for root, _, files in os.walk(folder_path) for f in files if f.endswith('.npy')]
        print(f"Total .npy files: {len(npy_files)}")
        for root, _, files in os.walk(folder_path):
            print(f"Files found: {len(files)} in {root}", flush=True)
            for file in files:
                
                
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        data = np.load(file_path, allow_pickle=True).item()

                        if 'rain_rate' not in data:
                            continue

                        r = np.array(data['rain_rate'], dtype=np.float32)
                        if r.ndim > 2:
                            r = r.squeeze()
                        if r.ndim != 2:
                            continue  # skip non-2D arrays

                        count = number_elements_in_image(r, threshold)
                        distribution[file] = count

                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    print(f"Total number of images processed: {len(distribution)}")

    # Plot histogram
    counts = list(distribution.values())
    if counts:
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=bins, color='skyblue', edgecolor='black')
        plt.xlabel(f"Number of elements > {threshold}")
        plt.ylabel("Frequency")
        plt.title(title or f"Distribution of number of elements > {threshold}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            print(f"Histogram of element counts saved to {output_path}")
        if show:
            plt.show()
        else:
            plt.close()

    return distribution


if __name__ == "__main__":
        base_path = '/net/nfs/ssd3/cfrancoismartin/Projects/datasets/fused_dataset/dataset'
        years = [str(year) for year in range(2008, 2024)]
        folders = [os.path.join(base_path, year) for year in years]
        # custom_bins = [
        #     (-100.0,0.0),
        #     (0.0, 0.1),
        #     (0.1, 0.5),
        #     (0.5, 1.0),
        #     (1.0, 5.0),
        #     (5.0, 10.0),
        #     (10.0, 50.0),
        #     (50.0, 100.0),
        #     (100.0,300.0)
        # ]
        # mean, std = get_mean_std_rain_rate(folders,exclude_zeros=True)
        # min_value, max_value = get_min_max_rain_rate(folders)
        # log_min_max_norm = lambda x: (np.log1p(x) - np.log1p(min_value)) / (np.log1p(max_value) - np.log1p(min_value))
        # log_mean_std_norm = lambda x: (np.log1p(x) - np.log1p(mean)) / (np.log1p(std))
        # plot_rain_rate_distribution(
        #     folder_paths=folders,
        #     output_path='/net/nfs/ssd3/cfrancoismartin/Projects/datasets/results/rain_rate_distribution_no_transform.png',
        #     custom_bins=None,
        #     log_scale=False,
        #     show=True,
        #     transform_fn=None,
        #     title="Distribution of 'rain_rate' Values (no transform)")
        
        # print("Rain rate distribution plot generated successfully.")
        plot_distribution_of_nb_elements(
            folder_paths=folders,
            threshold=0.1,
            output_path='/net/nfs/ssd3/cfrancoismartin/Projects/datasets/results/new_distribution_of_nb_elements.png',
            show=True)
        # count_rain_rate_shapes(
        #     folder_paths=folders,
        #     transform_fn=None,
        #     verbose=True
        # )