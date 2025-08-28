import os
import numpy as np
import matplotlib.pyplot as plt


def get_nan_counts_rain_rate(folder_paths):
    """
    Loads .npy files and returns the number of NaNs in the 'rain_rate' field per file.

    Parameters:
        folder_paths (list of str): List of folders to scan for .npy files.

    Returns:
        nan_counts (list of int): Number of NaNs in 'rain_rate' per file.
        file_paths (list of str): Corresponding list of file paths processed.
    """
    nan_counts = []
    file_paths = []

    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    full_path = os.path.join(root, file)
                    try:
                        data = np.load(full_path, allow_pickle=True).item()

                        r = np.array(data['rain_rate'], dtype=np.float16)
                        nan_count = np.isnan(r).sum()

                        nan_counts.append(nan_count)
                        file_paths.append(full_path)
                    except KeyError as e:
                        print(f"Skipping {full_path}: missing key {e}")
                    except Exception as e:
                        print(f"Error loading {full_path}: {e}")
    #print the percentage of image with less that 10% of nan values
    total_files = len(nan_counts)
    if total_files > 0:
        percentage = (np.array(nan_counts) < 0.05 * r.size).sum() / total_files * 100
        print(f"Percentage of files with less than 5% NaN values: {percentage:.2f}%")
    else:
        print("No files processed.")
    
    return nan_counts, file_paths

import matplotlib.pyplot as plt
import numpy as np

def plot_nan_distribution(nan_counts, save_path='nan_distribution.png', bins=30):
    """
    Plots and saves a histogram of NaN counts.

    Parameters:
        nan_counts (list or array): List of NaN counts per file.
        save_path (str): File path to save the histogram image.
        bins (int): Number of bins in the histogram.
    """
    if len(nan_counts) == 0:
        print("No NaN counts to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(nan_counts, bins=bins, color='skyblue', edgecolor='black')
    #plt.yscale('log')
    plt.title("Distribution of NaN Values in 'rain_rate'")
    plt.xlabel("Number of NaN Values")
    plt.ylabel("Number of Files (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Histogram saved to: {save_path}")

def plot_max_vs_nan_scatter(folder_paths, save_path='max_vs_nan_scatter.png'):
    """
    Plots and saves a scatter plot of max(rain_rate) vs. number of NaNs in rain_rate for each .npy file.

    Parameters:
        folder_paths (list of str): List of folders to scan for .npy files.
        save_path (str): File path to save the scatter plot image.
    """
    max_values = []
    nan_counts = []
    file_paths = []

    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    full_path = os.path.join(root, file)
                    try:
                        data = np.load(full_path, allow_pickle=True).item()
                        r = np.array(data['rain_rate'], dtype=np.float32)

                        nan_count = np.isnan(r).sum()
                        max_val = np.nanmax(r)

                        nan_counts.append(nan_count)
                        max_values.append(max_val)
                        file_paths.append(full_path)

                    except KeyError as e:
                        print(f"Skipping {full_path}: missing key {e}")
                    except Exception as e:
                        print(f"Error loading {full_path}: {e}")

    if not max_values:
        print("No valid data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(max_values, nan_counts, alpha=0.6, color='teal', edgecolor='black', linewidth=0.3)
    plt.title("Scatter Plot of Max Rain Rate vs. Number of NaN Values")
    plt.xlabel("Max Rain Rate (ignoring NaNs)")
    plt.ylabel("Number of NaN Values")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Scatter plot saved to: {save_path}")



def plot_max_vs_nan_heatmap(folder_paths, save_path='max_vs_nan_heatmap.png', bins=(100, 100)):
    """
    Plots and saves a 2D histogram (heatmap) of max(rain_rate) vs. number of NaNs in rain_rate.

    Parameters:
        folder_paths (list of str): List of folders to scan for .npy files.
        save_path (str): File path to save the heatmap image.
        bins (tuple): Number of bins for (x, y) axes in the 2D histogram.
    """
    max_values = []
    nan_counts = []

    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    full_path = os.path.join(root, file)
                    try:
                        data = np.load(full_path, allow_pickle=True).item()
                        r = np.array(data['rain_rate'], dtype=np.float32)

                        nan_count = np.isnan(r).sum()
                        max_val = np.nanmax(r)

                        nan_counts.append(nan_count)
                        max_values.append(max_val)

                    except KeyError as e:
                        print(f"Skipping {full_path}: missing key {e}")
                    except Exception as e:
                        print(f"Error loading {full_path}: {e}")

    if not max_values:
        print("No valid data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist2d(max_values, nan_counts, bins=bins, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(label='Number of Files (log scale)')
    plt.xlabel("Max Rain Rate (ignoring NaNs)")
    plt.ylabel("Number of NaN Values")
    plt.title("Heatmap of Max Rain Rate vs. NaN Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Heatmap saved to: {save_path}")

    
if __name__ == "__main__":
    base_path = '/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset'
    years = [str(year) for year in range(2008, 2024)]
    folders = [os.path.join(base_path, year) for year in years]
    # Example usage
    nan_counts, files = get_nan_counts_rain_rate(folders)
    # plot_max_vs_nan_heatmap(folders, save_path='/net/nfs/ssd3/cfrancoismartin/Projects/datasets/max_vs_nan_heatmap2.png', bins=(100, 100))
    plot_nan_distribution(nan_counts, save_path='/net/nfs/ssd3/cfrancoismartin/Projects/datasets/3_rain_rate_nan_distribution_done.png', bins=100)
    # plot_max_vs_nan_scatter(folders, save_path='/net/nfs/ssd3/cfrancoismartin/Projects/datasets/max_vs_nan_scatter2.png')