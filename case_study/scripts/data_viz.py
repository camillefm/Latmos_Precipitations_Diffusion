import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import pandas as pd
import torch
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm


from core.src.Visualization.plot import plot_heatmap, plot_histogram


data_dir = "/net/nfs/ssd3/cfrancoismartin/Projects/ClimMatthieu/data"
output_dir = "/net/nfs/ssd3/cfrancoismartin/Projects/ClimMatthieu/results"
input_list = ['IR_087','IR_108','IR_120','WV_062','WV_073','IR_134','IR_097']
filename = "sevmos_2019-08-09_18:10:00.nc"

def get_pixel_values(tensor):
    tensor=tensor[~torch.isnan(tensor)]
    return tensor.detach().cpu().numpy().flatten().tolist()

def extraction(file,treshold = 0.01) :
    nc = netCDF4.Dataset(file) # reading the nc file and creating Dataset

    data = {
        'WV_062': nc['WV_062'][:],  # float32 (y, x)
        'WV_073': nc['WV_073'][:],  # float32 (y, x)
        'IR_087': nc['IR_087'][:],  # float32 (y, x)
        'IR_097': nc['IR_097'][:],  # float32 (y, x)
        'IR_108': nc['IR_108'][:],  # float32 (y, x)
        'IR_120': nc['IR_120'][:],  # float32 (y, x)
        'IR_134': nc['IR_134'][:],  # float32 (y, x)
        'rain_rate': nc['rain_rate'],  # float32 (x, y)
        'rain_quality': nc['rain_quality'][:]  # int32 (x, y)
    }

    #print(data)
    lower, upper = data['rain_rate'].valid_range
    fill_value = data['rain_rate'].missing_value
    
    # Ensure valid_range is compatible with the rain_rate type (convert to float if needed)
    lower = float(lower)  # Convert to float to match the rain_rate's type (float32)
    upper = float(upper)  # Convert to float to match the rain_rate's type (float32)
    
    # Extract rain_rate data and apply the valid_range filter
    rain_rate =  data['rain_rate'][:]

    # Apply the valid range filter: values outside this range will be replaced with NaN
    data['rain_rate'] = np.where((rain_rate >= 0.0) & (rain_rate <= upper), rain_rate,np.nan)

    nc.close()
    #data = np.array(data)
    return data 

def extract_npy(fp):
    """
    Load the .npy file while allowing object arrays.
    
    Parameters:
    - fp: str, path to the .npy file

    Returns:
    - data: Loaded numpy array or dictionary
    """

    return np.load(fp, allow_pickle=True).item()


def radar_quality(data,quality_treshold = 0 ):
        data['rain_rate'] = np.where(data['rain_quality']< quality_treshold,np.nan, data['rain_rate'])
        data['rain_rate'] = np.where(data['rain_quality']> 100,np.nan, data['rain_rate'])
        return data

def normalisation(data):
		"""
		Precomputed constant of normalisation for IR:
				mean            std 
		IR_087 	269.639838    	15.391083
		IR_108 	271.377943    	16.292396
		IR_120 	270.005003    	16.286951
		Normalisation for rain_rate:
		log(1+rr/rr0) avec rr0 = 1mm/h
		"""
		normalised_data = {
			'IR_087': (data['IR_087'] - 269.639838 )/ 15.391083,
			'IR_108': (data['IR_108'] - 271.377943 )/ 16.292396,
			'IR_120': (data['IR_120'] - 270.005003 )/ 16.286951,
                        'rain_rate': np.log1p(data['rain_rate']),			
                        'rain_quality': data['rain_quality']
		}
                
		return normalised_data

def plot_tb_rain_quality(data_dir, input_list, filename, result_dir):
    # Adjust if your utility module is elsewhere

    file_path = os.path.join(data_dir, filename)
    
    # Extract and process data
    data = extraction(file_path)
    data = radar_quality(data)

    # Grab the first IR channel
    first_tb_name = input_list[0]
    tb = data[first_tb_name]
    rain_rate = data['rain_rate']
    rain_quality = data['rain_quality']
    # Avoid log(0) in LogNorm
    rain_rate_safe = np.where(rain_rate>0.1,rain_rate,np.nan)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[1].set_aspect('equal')

    im0 = axes[0].imshow(tb, cmap='inferno')
    axes[0].set_title(f"TB Channel: {first_tb_name}")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(rain_rate_safe, cmap='jet', norm=LogNorm(vmin=0.1, vmax=50))
    axes[1].set_title("Rain Rate (Log Scale)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(rain_quality, cmap='viridis', vmin=0, vmax=100)
    axes[2].set_title("Rain Quality")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()

    # Safe filename and save
    filename_safe = filename.replace(":", "_").replace(".nc", "") + ".png"
    save_path = os.path.join(result_dir, filename_safe)
    plt.savefig(save_path,dpi=300)
    print(f"✅ Plot saved at {save_path}")

    plt.close()



def pad_to_height(img_arr, target_height):
    """Pads an image array vertically to match target height."""
    current_height = img_arr.shape[0]
    if current_height < target_height:
        padding = target_height - current_height
        pad_top = padding // 2
        pad_bottom = padding - pad_top
        return np.pad(img_arr, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=255)
    return img_arr

def plot_histogram_distribution(rain, sampled, png_path, cnn_sampled=None):
    """
    Save histogram of RainDiffusion vs sampled values, and optionally CNN values.

    Parameters:
        rain: tensor or array
        sampled: tensor or array
        png_path: str
        cnn_metric: tensor or array, optional
    """
    rain_pixel_values = get_pixel_values(rain)
    sampled_pixel_values = get_pixel_values(sampled)
    print(len(rain_pixel_values), len(sampled_pixel_values), min(rain_pixel_values),max(rain_pixel_values))
    image1 = plot_histogram(rain_pixel_values, sampled_pixel_values)
    img_arr1 = np.array(Image.fromarray(image1))

    if cnn_sampled is not None:
        cnn_pixel_values = get_pixel_values(cnn_sampled)
        image2 = plot_histogram(rain_pixel_values, cnn_pixel_values)
        img_arr2 = np.array(Image.fromarray(image2))

        # Ensure same height by padding smaller image
        max_height = max(img_arr1.shape[0], img_arr2.shape[0])
        img_arr1 = pad_to_height(img_arr1, max_height)
        img_arr2 = pad_to_height(img_arr2, max_height)

        fused = np.concatenate((img_arr1, img_arr2), axis=1)
    else:
        fused = img_arr1

    img = Image.fromarray(fused)
    img.save(png_path)
    print(f"Histogram saved to {png_path}")


def plot_confusion_matrix(range_confusions, png_path, cnn_metric=None):
    """
    Save confusion matrix (nested dict) as heatmap, optionally with CNN side-by-side.

    Parameters:
        range_confusions: dict[str, dict[str, float]]
        png_path: str
        cnn_metric: dict[str, dict[str, float]], optional
    """
    image1 = plot_heatmap(range_confusions, title="RainDiffusion", cmap="Blues", fmt=".2f")
    img_arr1 = np.array(Image.fromarray(image1))

    if cnn_metric is not None:
        image2 = plot_heatmap(cnn_metric, title="CNN", cmap="Blues", fmt=".2f")
        img_arr2 = np.array(Image.fromarray(image2))

        max_height = max(img_arr1.shape[0], img_arr2.shape[0])
        img_arr1 = pad_to_height(img_arr1, max_height)
        img_arr2 = pad_to_height(img_arr2, max_height)

        fused = np.concatenate((img_arr1, img_arr2), axis=1)
    else:
        fused = img_arr1

    img = Image.fromarray(fused)
    img.save(png_path)
    print(f"Confusion matrix saved to {png_path}")


def plot_range_metrics(range_metrics, png_path, cnn_metric=None):
    """
    Save range metrics (nested dict) as PNG, optionally comparing with CNN.

    Parameters:
        range_metrics: dict[str, dict[str, float]]
        png_path: str
        cnn_metric: dict[str, dict[str, float]], optional
    """
    all_metrics = sorted({metric for m in range_metrics.values() for metric in m.keys()})
    df = pd.DataFrame.from_dict(range_metrics, orient="index")
    df = df.reindex(columns=all_metrics)
    df.index.name = "rain_type"
    df.reset_index(inplace=True)

    if cnn_metric:
        df_cnn = pd.DataFrame.from_dict(cnn_metric, orient="index").reindex(columns=all_metrics)
        df_cnn.index.name = "rain_type"
        df_cnn.reset_index(inplace=True)

        for col in all_metrics:
            df[f"CNN_{col}"] = df_cnn[col]

    df[df.columns[1:]] = df[df.columns[1:]].round(3)

    fig, ax = plt.subplots(figsize=(1.5 * len(df.columns), 0.5 * len(df) + 1))
    ax.axis("off")
    mpl_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    mpl_table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Range metrics saved to {png_path}")

def log_metrics(metrics_dict, png_path, cnn_metric=None):
    """
    Save metrics dictionary as a PNG table, optionally comparing with CNN metrics.

    Parameters:
        metrics_dict: dict[str, float]
        png_path: str, path to save PNG file
        cnn_metric: dict[str, float], optional
    """
    if cnn_metric:
        df = pd.DataFrame({
            "Metric": list(metrics_dict.keys()),
            "RainDiffusion": list(metrics_dict.values()),
            "CNN": [cnn_metric.get(k, None) for k in metrics_dict.keys()]
        })
    else:
        df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])

    fig, ax = plt.subplots(figsize=(1.5 * len(df.columns), 0.5 * len(df) + 1))
    ax.axis("off")
    mpl_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    mpl_table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Metrics table saved to {png_path}")


def plot_difference(diffusion_image, target, png_path, cnn_image=None):
    """
    Plots a grid:
    Row 1: Target | Diffusion | Diffusion - Target
    Row 2 (optional): Target | CNN | CNN - Target
    Raw images use log scale (0.1 to 50) with 'jet' colormap.
    Differences: red = negative, blue = positive, white = zero.
    """
    def to_gray(img):
        """Convert tensor to grayscale numpy array."""
        if img.ndim == 3:  # (C, H, W) → grayscale
            img = img.mean(dim=0)
        return img.detach().cpu().numpy()
   
    def compute_diff(img):
        return to_gray(target)-to_gray(img)
   
    # Convert images to grayscale numpy
    target_np = to_gray(target)
    diff_np = to_gray(diffusion_image)
    diff_diff_np = compute_diff(diffusion_image)
   
    cnn_np = None
    diff_cnn_np = None
    if cnn_image is not None:
        cnn_np = to_gray(cnn_image)
        diff_cnn_np = compute_diff(cnn_image)
   
    # Symmetric log scale for differences (-200 to 200)
    symlog_norm_diff = SymLogNorm(linthresh=1, linscale=1, vmin=-50, vmax=50)
   
    # Figure setup
    rows = 2 if cnn_image is not None else 1
    fig, axes = plt.subplots(rows, 3, figsize=(12, 5 * rows))
    if rows == 1:
        axes = [axes]  # ensure iterable
   
    log_norm = LogNorm(vmin=0.1, vmax=50)
   
    # Row 1: Target | Diffusion | Diffusion - Target
    axes[0][0].imshow(target_np, cmap="jet", norm=log_norm)
    axes[0][0].set_title("Target")
    axes[0][0].axis("off")
    
    diff_np[diff_np < 0.1] = np.nan
    axes[0][1].imshow(diff_np, cmap="jet", norm=log_norm)
    axes[0][1].set_title("Diffusion")
    axes[0][1].axis("off")
   
    im_diff = axes[0][2].imshow(diff_diff_np, cmap="bwr", norm=symlog_norm_diff)
    axes[0][2].set_title("Target - Diffusion")
    axes[0][2].axis("off")
   
    # Row 2: Target | CNN | CNN - Target
    if cnn_image is not None:
        axes[1][0].imshow(target_np, cmap="jet", norm=log_norm)
        axes[1][0].set_title("Target")
        axes[1][0].axis("off")
       
        axes[1][1].imshow(cnn_np, cmap="jet", norm=log_norm)
        axes[1][1].set_title("CNN")
        axes[1][1].axis("off")
       
        axes[1][2].imshow(diff_cnn_np, cmap="bwr", norm=symlog_norm_diff)
        axes[1][2].set_title("Target - CNN")
        axes[1][2].axis("off")
   
    # Shared colorbars
    raw_axes = []
    diff_axes = []
   
    for i in range(rows):
        raw_axes.extend([axes[i][0], axes[i][1]])  # Target and prediction images
        diff_axes.append(axes[i][2])  # Difference images
   
    fig.colorbar(plt.cm.ScalarMappable(norm=log_norm, cmap="jet"),
                 ax=raw_axes,
                 orientation="vertical", fraction=0.046, pad=0.04, label="Value (log scale)")
   
    fig.colorbar(im_diff, ax=diff_axes,
                 orientation="vertical", fraction=0.046, pad=0.04, label="Difference (symmetric log scale)")
   
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()

def plot_differences_only(diffusion_image, target, png_path, cnn_image=None):
    """
    Plots only the differences side by side:
    - Diffusion - Target (left)
    - CNN - Target (right, if CNN provided)
    
    Uses red/blue colormap where red = negative, blue = positive, white = zero.
    Each subplot has its own colorbar positioned to the right.
    """
    def to_gray(img):
        """Convert tensor to grayscale numpy array."""
        if img.ndim == 3:  # (C, H, W) → grayscale
            img = img.mean(dim=0)
        return img.detach().cpu().numpy()
   
    def compute_diff(img):
        return to_gray(target) - to_gray(img)
   
    # Compute differences
    diff_diffusion = compute_diff(diffusion_image)
    diff_cnn = None
    if cnn_image is not None:
        diff_cnn = compute_diff(cnn_image)
    
    # Determine number of subplots
    num_plots = 2 if cnn_image is not None else 1
    
    # Create figure with appropriate width
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]  # Ensure iterable
    
    # Symmetric log scale for differences
    symlog_norm = SymLogNorm(linthresh=1, linscale=1, vmin=-50, vmax=50)
    
    # Plot Diffusion difference
    im1 = axes[0].imshow(diff_diffusion, cmap="bwr", norm=symlog_norm)
    axes[0].set_title("Target - Diffusion")
    axes[0].axis("off")
    
    # Add colorbar for diffusion difference
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Difference (symmetric log scale)")
    
    # Plot CNN difference if provided
    if cnn_image is not None:
        im2 = axes[1].imshow(diff_cnn, cmap="bwr", norm=symlog_norm)
        axes[1].set_title("Target - CNN")
        axes[1].axis("off")
        
        # Add colorbar for CNN difference
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label("Difference (symmetric log scale)")
    
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()


import io

def plot_density_heatmap(rain, sampled, png_path, cnn_sampled=None, bins=100, figsize=(12, 5)):
    """
    Save density heatmap of target vs sampled values, and optionally target vs CNN values.
    
    Parameters:
        rain: tensor or array (target values, may contain NaN)
        sampled: tensor or array (sampled values)
        png_path: str (path to save the image)
        cnn_sampled: tensor or array, optional (CNN sampled values)
        bins: int, number of bins for the 2D histogram
        figsize: tuple, figure size for each subplot
    """
    
    # Convert to numpy arrays and flatten
    rain_flat = rain.detach().cpu().numpy().flatten()
    sampled_flat = sampled.detach().cpu().numpy().flatten()
    
    # Create mask for non-NaN values in rain
    valid_mask = ~np.isnan(rain_flat)
    
    # Filter out NaN pixels
    rain_valid = rain_flat[valid_mask]
    sampled_valid = sampled_flat[valid_mask]
    
    print(f"Valid pixels: {len(rain_valid)} out of {len(rain_flat)}")
    print(f"Rain range: [{np.min(rain_valid):.3f}, {np.max(rain_valid):.3f}]")
    print(f"Sampled range: [{np.min(sampled_valid):.3f}, {np.max(sampled_valid):.3f}]")
    
    # Determine number of subplots
    n_plots = 2 if cnn_sampled is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0] * n_plots, figsize[1]))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Target vs Sampled
    hist1, xedges, yedges = np.histogram2d(rain_valid, sampled_valid, bins=bins)
    # Add small constant to avoid log(0)
    hist1 = hist1 + 1e-10
    
    im1 = axes[0].imshow(hist1.T, origin='lower', 
                         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                         aspect='auto', cmap='tab20c_r', norm=LogNorm(vmin=1e-1))
    axes[0].plot([np.min(rain_valid), np.max(rain_valid)], 
                 [np.min(rain_valid), np.max(rain_valid)], 
                 'r--', alpha=0.7, linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('Target (Rain)')
    axes[0].set_ylabel('Sampled')
    
    axes[0].set_title('Target vs Sampled Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0], label='Density (log scale)')
    
    # Plot 2: Target vs CNN (if provided)
    if cnn_sampled is not None:
        cnn_flat = cnn_sampled.detach().cpu().numpy().flatten()
        cnn_valid = cnn_flat[valid_mask]
        
        print(f"CNN range: [{np.min(cnn_valid):.3f}, {np.max(cnn_valid):.3f}]")
        
        hist2, xedges2, yedges2 = np.histogram2d(rain_valid, cnn_valid, bins=bins)
        hist2 = hist2 + 1e-10
        
        im2 = axes[1].imshow(hist2.T, origin='lower',
                            extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]],
                            aspect='auto', cmap='tab20c_r', norm=LogNorm(vmin=1e-1))
        axes[1].plot([np.min(rain_valid), np.max(rain_valid)], 
                     [np.min(rain_valid), np.max(rain_valid)], 
                     'r--', alpha=0.7, linewidth=2, label='Perfect prediction')
        axes[1].set_xlabel('Target (Rain)')
        axes[1].set_ylabel('CNN Sampled')
        axes[1].set_title('Target vs CNN Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(im2, ax=axes[1], label='Density (log scale)')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Density heatmap saved to {png_path}")


def plot_density_heatmap_combined(rain, sampled, png_path, cnn_sampled=None, bins=50):
    """
    Alternative version that returns a PIL Image array similar to the original function structure.
    """
    # Convert to numpy arrays and flatten
    rain_flat = np.array(rain).flatten()
    sampled_flat = np.array(sampled).flatten()
    
    # Create mask for non-NaN values in rain
    valid_mask = ~np.isnan(rain_flat)
    
    # Filter out NaN pixels
    rain_valid = rain_flat[valid_mask]
    sampled_valid = sampled_flat[valid_mask]
    
    def create_heatmap_image(x_data, y_data, title, bins=bins):
        fig, ax = plt.subplots(figsize=(6, 5))
        hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)
        hist = hist + 1e-10  # Avoid log(0)
        
        im = ax.imshow(hist.T, origin='lower',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       aspect='auto', cmap='tab20c_r', norm=LogNorm(vmin=1e-1))
        ax.plot([np.min(x_data), np.max(x_data)], 
                [np.min(x_data), np.max(x_data)], 
                'r--', alpha=0.7, linewidth=2)
        ax.set_xlabel('Target (Rain)')
        ax.set_ylabel(title.split(' vs ')[-1])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, label='Density')
        
        # Convert to image array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        img_arr = np.array(img)
        plt.close()
        buf.close()
        
        return img_arr
    
    # Create first heatmap
    img_arr1 = create_heatmap_image(rain_valid, sampled_valid, "Target vs Sampled")
    
    # Create second heatmap if CNN data provided
    if cnn_sampled is not None:
        cnn_flat = cnn_sampled.detach().cpu().numpy().flatten()
        cnn_valid = cnn_flat[valid_mask]
        img_arr2 = create_heatmap_image(rain_valid, cnn_valid, "Target vs CNN")
        
        # Ensure same height by padding smaller image
        max_height = max(img_arr1.shape[0], img_arr2.shape[0])
        if img_arr1.shape[0] < max_height:
            pad_height = max_height - img_arr1.shape[0]
            img_arr1 = np.pad(img_arr1, ((0, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=255)
        elif img_arr2.shape[0] < max_height:
            pad_height = max_height - img_arr2.shape[0]
            img_arr2 = np.pad(img_arr2, ((0, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=255)
        
        # Concatenate horizontally
        fused = np.concatenate((img_arr1, img_arr2), axis=1)
    else:
        fused = img_arr1
    
    # Save the combined image
    img = Image.fromarray(fused.astype(np.uint8))
    img.save(png_path)
    print(f"Density heatmap saved to {png_path}")
    
    return fused