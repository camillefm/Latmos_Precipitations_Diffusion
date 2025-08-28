import torch
import random
from torch.utils.data import DataLoader
import csv
import numpy as np

class SharedRandomCrop3D_XYRQ:
    """
    Applies the same random crop to x (input), y (target), and rq (rain_quality).
    Assumes all are 3D tensors of shape (C, H, W) or (H, W).
    """
    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_h = self.crop_w = crop_size
        else:
            self.crop_h, self.crop_w = crop_size

    def __call__(self, x, y, rq):
        # Ensure all are tensors
        _, h, w = x.shape

        if h < self.crop_h or w < self.crop_w:
            raise ValueError(f"Crop size ({self.crop_h},{self.crop_w}) must be <= input size ({h},{w})")

        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)

        # Crop all three
        x_cropped = x[:, top:top + self.crop_h, left:left + self.crop_w]
        y_cropped = y[:, top:top + self.crop_h, left:left + self.crop_w] if y.ndim == 3 else y
        rq_cropped = rq[:, top:top + self.crop_h, left:left + self.crop_w] if rq.ndim == 3 else rq

        return x_cropped, y_cropped, rq_cropped


class MinMaxNormalization:
    """
    Normalize input x (C, H, W) to the range [0, 1].

    """
    def __init__(self):
        pass

    def __call__(self, x):
        # Normalize to range [0, 1]
        return (x - x.min()) / (x.max() - x.min()).to(dtype=torch.float32)
    
class MSNormalization:
    """
    Normalize each channel of input tensor x using per-channel mean and std.
    Input can be (C, H, W) or (N, C, H, W).

    """
    def __init__(self, mean, std, nb_of_sigmas=1):
        mean = torch.tensor(mean, dtype=torch.float32) if not isinstance(mean, torch.Tensor) else mean
        std = torch.tensor(std, dtype=torch.float32) if not isinstance(std, torch.Tensor) else std

        # Reshape for broadcasting: (C, 1, 1)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
        self.nb_of_sigmas = nb_of_sigmas

    def __call__(self, x):
        # Check if input is batched: (N, C, H, W)
        if x.dim() == 4:
            mean = self.mean.unsqueeze(0)  # (1, C, 1, 1)
            std = self.std.unsqueeze(0)
        else:  # (C, H, W)
            mean = self.mean
            std = self.std

        return ((x - mean) / (std * self.nb_of_sigmas)).to(dtype=torch.float32)

class LogNormalisation:
    """ 
    Normalizes the input using log(1+x) transformation.
    """ 
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, x):
        return torch.log1p(x + self.eps).to(dtype=torch.float32)
    

def estimate_mean_std(dataset, num_samples=100, data_key='tb', exclude_zeros=False):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    mean = 0.0
    std = 0.0
    count = 0

    for i, (tb, r, rq) in enumerate(loader):
        if i >= num_samples:
            break
        data = {'tb': tb, 'r': r, 'rq': rq}[data_key].float()
        if exclude_zeros:
            nonzero_mask = data >= 0.1
            nonzero_data = data[nonzero_mask]

            if nonzero_data.numel() <= 1:
                continue  # Skip if 0 or 1 elements

            sample_mean = nonzero_data.mean()
            sample_std = nonzero_data.std()
        else:
            if data.numel() <= 1:
                continue  # Also skip tiny tensors when not excluding zeros

            sample_mean = data.mean(dim=[2, 3]) if data.ndim == 4 else data.mean()
            sample_std = data.std(dim=[2, 3]) if data.ndim == 4 else data.std()

        mean += sample_mean
        std += sample_std
        count += 1

    if count == 0:
        raise ValueError("No valid (non-trivial) data found in the sampled batches.")

    mean /= count
    std /= count

    return mean, std

def save_mean_std_to_csv(filename, mean, std):
    """
    Saves mean and std tensors/lists/arrays into a CSV file with two columns: 'mean' and 'std'.
    Each cell will contain a list (as a string).

    Args:
        filename (str): Path to the CSV file.
        mean: Tensor, list, or ndarray representing mean values.
        std: Tensor, list, or ndarray representing std values.
    """

    # Convert to lists if needed
    def to_list(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().tolist()
        elif isinstance(x, np.ndarray):
            return x.tolist()
        return x  # assume it's already a list or scalar

    mean_list = to_list(mean)
    std_list = to_list(std)

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mean', 'std'])
        writer.writerow([mean_list, std_list])
    
