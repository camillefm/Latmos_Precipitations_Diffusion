import os
import numpy as np
from scipy import interpolate

def interpolate_nans(arr, method='linear'):
    """
    Interpolates NaN values in 1D, 2D, or 3D numpy arrays.
    If 'linear' interpolation fails or leaves NaNs, fallback to 'nearest'.

    Args:
        arr (np.ndarray): Input array containing NaNs.
        method (str): Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        np.ndarray: Interpolated array (may still contain NaNs if interpolation fails).
    """
    if not np.isnan(arr).any():
        return arr

    arr = np.array(arr, dtype=np.float32)

    if arr.ndim == 1:
        x = np.arange(len(arr))
        mask = ~np.isnan(arr)
        if mask.sum() < 2:
            print("Too few valid points for 1D interpolation")
            return arr
        return np.interp(x, x[mask], arr[mask])

    elif arr.ndim == 2:
        x = np.arange(arr.shape[1])
        y = np.arange(arr.shape[0])
        xx, yy = np.meshgrid(x, y)
        mask = ~np.isnan(arr)

        if np.sum(mask) < 4:
            print("Too few valid points for 2D interpolation")
            return arr

        # First attempt: use specified method
        result = interpolate.griddata(
            (yy[mask], xx[mask]),
            arr[mask],
            (yy, xx),
            method=method,
            fill_value=np.nan
        )

        # Fallback: try 'nearest' if NaNs remain and method wasn't already 'nearest'
        if np.isnan(result).any() and method != 'nearest':
            print("NaNs remain after", method, "interpolation. Retrying with 'nearest'.")
            result = interpolate.griddata(
                (yy[mask], xx[mask]),
                arr[mask],
                (yy, xx),
                method='nearest',
                fill_value=np.nan
            )

        return result

    elif arr.ndim == 3:
        # Process each 2D slice independently
        return np.stack([interpolate_nans(slice_, method) for slice_ in arr])

    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")

import os
import numpy as np

def remove_low_rain_files(
    folder_paths, 
    min_rain_percent=1.0, 
    delete=False, 
    dry_run=False, 
    backup_original=False
):
    """
    Removes .npy files where 'rain_rate' has less than a threshold percentage of rain pixels (>0.1).

    Parameters:
        folder_paths (list of str): Folders to search for .npy files.
        min_rain_percent (float): Minimum required percent of rain pixels (>0.1).
        delete (bool): If True, delete files falling below the threshold.
        dry_run (bool): If True, simulate actions without modifying any files.
        backup_original (bool): If True, make a backup before deleting or modifying.

    Returns:
        removed_files (list): List of deleted or flagged files.
    """
    removed_files = []

    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        data = np.load(file_path, allow_pickle=True).item()

                        if 'rain_rate' not in data:
                            print(f"Skipping {file_path}: 'rain_rate' not found.")
                            continue

                        r = np.array(data['rain_rate'], dtype=np.float32)
                        total = r.size
                        rain_pixels = (r > 0.1).sum()
                        rain_percent = (rain_pixels / total) * 100

                        if rain_percent < min_rain_percent:
                            removed_files.append(file_path)
                            if delete:
                                if not dry_run:
                                    if backup_original:
                                        os.rename(file_path, file_path + ".bak")
                                    os.remove(file_path)
                                print(f"Deleted {file_path} ({rain_percent:.2f}% rain)")
                            else:
                                print(f"Would delete {file_path} ({rain_percent:.2f}% rain)")

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    print("\n=== Summary ===")
    print(f"Total files deleted/flagged: {len(removed_files)}")
    return removed_files

def remove_or_interpolate_nan_files(
    folder_paths, 
    max_nan_percent=5.0, 
    delete=False, 
    interp_method='linear',
    dry_run=False,
    backup_original=False
):
    """
    Removes or interpolates .npy files based on the percentage of NaNs in 'rain_rate'.

    Parameters:
        folder_paths (list of str): Folders to search for .npy files.
        max_nan_percent (float): Threshold of NaNs to delete. Below this, NaNs are interpolated.
        delete (bool): If True, delete files exceeding threshold.
        interp_method (str): Interpolation method: 'linear', 'nearest', or 'cubic'.
        dry_run (bool): If True, simulate actions without modifying any files.
        backup_original (bool): If True, make a backup of the original file before overwriting.

    Returns:
        removed_files (list): List of deleted files.
        interpolated_files (list): List of files with interpolated NaNs.
    """
    removed_files = []
    interpolated_files = []

    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        data = np.load(file_path, allow_pickle=True).item()

                        if 'rain_rate' not in data:
                            print(f"Skipping {file_path}: 'rain_rate' not found.")
                            continue

                        r = np.array(data['rain_rate'], dtype=np.float32)
                        total = r.size
                        nan_count = np.isnan(r).sum()
                        nan_percent = (nan_count / total) * 100

                        if nan_percent > max_nan_percent:
                            removed_files.append(file_path)
                            if delete:
                                if not dry_run:
                                    os.remove(file_path)
                                print(f"Deleted {file_path} ({nan_percent:.2f}% NaNs)")
                            else:
                                print(f"Would delete {file_path} ({nan_percent:.2f}% NaNs)")
                        else:
                            r_interp = interpolate_nans(r, method=interp_method)
                            new_nan_count = np.isnan(r_interp).sum()

                            if new_nan_count < nan_count:
                                data['rain_rate'] = r_interp
                                if not dry_run:
                                    if backup_original:
                                        os.rename(file_path, file_path + ".bak")
                                    np.save(file_path, data, allow_pickle=True)
                                interpolated_files.append(file_path)
                                print(f"Interpolated {file_path} ({nan_count} → {new_nan_count} NaNs)")
                            else:
                                print(f"No improvement after interpolation: {file_path} ({nan_count} NaNs)")

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    print("\n=== Summary ===")
    print(f"Total files deleted:      {len(removed_files)}")
    print(f"Total files interpolated: {len(interpolated_files)}")
    return removed_files, interpolated_files


# Example usage:
if __name__ == "__main__":
    folders = ["/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2008", "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2009",
               "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2010", "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2011",
               "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2012", "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2013",
               "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2014", "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2015",
              "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2016", "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2017",
              "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2018", # "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2019",
              "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2020", "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2021",
              "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2022", "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2023"]
    # remove_or_interpolate_nan_files(
    #     folders,
    #     max_nan_percent=5.0,
    #     delete=True,
    #     interp_method='linear',
    #     dry_run=False,
    #     backup_original=True
    # )

    remove_low_rain_files(
        folders,
        min_rain_percent=10.0,
        delete=True,
        dry_run=False,
        backup_original=False
    )