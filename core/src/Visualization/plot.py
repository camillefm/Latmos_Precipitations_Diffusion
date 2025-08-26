import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import torch
import io
from PIL import Image
import seaborn as sns # type: ignore
import pandas as pd
from matplotlib.ticker import LogLocator, ScalarFormatter


def standardize_tensor_shape(tensor, expect_4d=False):
    """
    Converts tensor to shape (B, H, W) or (B, C, H, W) if expect_4d is True.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        expect_4d (bool): Whether the output should be (B, C, H, W).
    
    Returns:
        torch.Tensor: Standardized tensor.
    """
    if tensor is None:
        return None

    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")

    # Convert to 4D if needed
    if expect_4d:
        if tensor.dim() == 4:
            return tensor
        elif tensor.dim() == 3:
            return tensor.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
        elif tensor.dim() == 2:
            return tensor.unsqueeze(0).unsqueeze(0)  # (H, W) → (1, 1, H, W)
        else:
            raise ValueError(f"Unsupported tensor shape {tensor.shape} for 4D expectation")
    else:
        # Expect 3D: (B, H, W)
        if tensor.dim() == 3:
            return tensor
        elif tensor.dim() == 4 and tensor.size(0) == 1:
            return tensor.squeeze(0)  # (1, B, H, W) → (B, H, W)
        elif tensor.dim() == 4 and tensor.size(1) == 1:
            return tensor.squeeze(1) # (B,1,H,W) → (B,H,W) 
        elif tensor.dim() == 2:
            return tensor.unsqueeze(0)  # (H, W) → (1, H, W)
        else:
            raise ValueError(f"Unsupported tensor shape {tensor.shape} for 3D expectation")
        
def plot_tensors(tb_tensor=None, rain_tensor=None, sampled_tensor=None, rq_tensor=None, tb_normalised_nb_sigmas = 1):
    vmin = 0.1
    vmax = 50
    log_norm = LogNorm(vmin=vmin, vmax=vmax)
    rq_norm = Normalize(vmin=0, vmax=100)
    
    if tb_normalised_nb_sigmas == 1:
        tb_norm = Normalize(vmin=-3, vmax=3)
    elif tb_normalised_nb_sigmas == 3:
        tb_norm = tb_norm = Normalize(vmin=-1, vmax=1)
    else :
        tb_norm = Normalize(vmin=200, vmax=300)


    tensor_list = []
    titles = []
    cmaps = []
    norms = []
    cols=0

    if tb_tensor is not None:
        tb_tensor = standardize_tensor_shape(tb_tensor, expect_4d=True)
        tensor_list.append(tb_tensor[:, 0, :, :].cpu())  # Select channel 0
        titles.append("tb")
        cmaps.append("inferno_r")
        norms.append(tb_norm)  # Linear scale for tb
        cols += 1
    if rain_tensor is not None:
        rain_tensor = standardize_tensor_shape(rain_tensor, expect_4d=False)
        tensor_list.append(rain_tensor.cpu())
        titles.append("rain")
        cmaps.append("jet")
        norms.append(log_norm)
        cols += 1

    if sampled_tensor is not None:
        sampled_tensor = standardize_tensor_shape(sampled_tensor, expect_4d=False)
        tensor_list.append(sampled_tensor.cpu())
        titles.append("sampled")
        cmaps.append("jet")
        norms.append(log_norm)
        cols += 1

    if rq_tensor is not None:
        rq_tensor = standardize_tensor_shape(rq_tensor, expect_4d=False)
        tensor_list.append(rq_tensor.cpu())
        titles.append("rq")
        cmaps.append("viridis")
        norms.append(rq_norm)  # Linear scale for rq
        cols += 1

    assert len(tensor_list) > 0, "At least one tensor must be provided for plotting."

    batch_size = tensor_list[0].shape[0]
    num_inputs = len(tensor_list)

    if cols == 1:
        cols = min(batch_size, 8)  # max 8 columns
        rows = (batch_size + cols - 1) // cols  # ceil division for rows
    else:
        rows = batch_size
        cols = num_inputs


    figsize = (cols * 4, rows * 4)
    total_plots = batch_size * num_inputs
    if rows * cols < total_plots:
        raise ValueError(f"Grid too small: rows*cols={rows*cols} < total plots {total_plots}")

    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for b in range(batch_size):
        for i, (tensor_np, title, cmap, norm) in enumerate(zip(tensor_list, titles, cmaps, norms)):
            plot_idx = b * num_inputs + i  # plots in order: 0,1,2,3, 4,5,6,7 ...
            r = plot_idx // cols
            c = plot_idx % cols

            ax = axs[r, c]
            im = ax.imshow(tensor_np[b], cmap=cmap, norm=norm)
            if r == 0:
                ax.set_title(title)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off unused subplots if any
    for idx in range(total_plots, rows * cols):
        r = idx // cols
        c = idx % cols
        axs[r, c].axis('off')

    plt.tight_layout()

    # Save to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf)
    return np.array(image)

def plot_histogram(rain_pixel_values, sampled_pixel_values):
    # Convert inputs to NumPy arrays
    rain_pixel_values = np.array(rain_pixel_values)
    sampled_pixel_values = np.array(sampled_pixel_values)

    # Fixed bin range in log space
    bins = np.logspace(np.log10(0.1), np.log10(350), 50)

    # Compute histograms
    hist1, _ = np.histogram(rain_pixel_values, bins=bins)
    hist2, _ = np.histogram(sampled_pixel_values, bins=bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    diff = hist2 - hist1

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- Top histogram plot ---
    ax1.hist(rain_pixel_values, bins=bins, alpha=0.5, label='Target Rain', color='green')
    ax1.hist(sampled_pixel_values, bins=bins, alpha=0.5, label='Sampled Rain', color='blue')
    ax1.set_xscale('log')
    ax1.set_xlim(0.1, 350)
    ax1.set_yscale('log')
    
    ax1.set_ylabel("Frequency")
    ax1.set_title("Rain VS Sampled Rain Pixel Values")
    ax1.legend()

    # --- Bottom difference plot ---
    ax2.set_xscale('log')
    ax2.set_xlim(0.1, 350)
    ax2.set_yscale('symlog', linthresh=1)
    ax2.set_ylim(-1e5, 1e5)
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Difference")
    ax2.set_title("Histogram Difference (Sampled - Target)")

    ax2.bar(bin_centers, diff, width=np.diff(bins), 
            color=np.where(diff > 0, 'green', 'red'), align='center')
    ax2.axhline(0, color='black', linestyle='--')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)

    return np.array(image)




def plot_heatmap(data_dict, title="Class Coincidence Heatmap", cmap="Blues", fmt=".2f"):
    """
    Plots a heatmap from a nested dictionary (dict of dicts).
    Outer keys: true classes
    Inner keys: predicted classes (or any metric keys)
    
    Returns:
        image (np.ndarray): The rendered heatmap as a NumPy array.
    """
    # Convert nested dictionary to DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient="index").fillna(0)


    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5, linecolor='black')
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()

    # Save to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    plt.close()

    return np.array(image)


if __name__ == "__main__":
    # # Example usage
    # tb_tensor = torch.rand(10,3, 64, 64) *50  # Example tensor
    # rain_tensor = torch.rand(10,1, 64, 64) * 50  # Example tensor
    # sampled_tensor = torch.rand(10,1,64, 64) * 50  # Example tensor
    # rq_tensor = torch.rand(10,1, 64, 64) * 50  # Example tensor

    # result_image = plot_tensors(tb_tensor)

    metrics_dict = {
    'no_rain': {'no_rain': 0.8, 'moderate_rain': 0.2},
    'moderate_rain': {'no_rain': 0.1, 'moderate_rain': 0.7, 'heavy_rain': 0.2},
    'heavy_rain': {'moderate_rain': 0.3, 'heavy_rain': 0.7}
}

    result_image = plot_heatmap(metrics_dict, title="Rain Diffusion Confusion Metrics Heatmap")
    plt.imshow(result_image)
    plt.show()
 
    
    Image.fromarray(result_image).save("output_image.png")

    plt.imshow(result_image)
    plt.axis('off')
    plt.show()