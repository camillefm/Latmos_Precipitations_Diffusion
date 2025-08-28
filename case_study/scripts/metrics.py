#9 aout 2019
import sys
import os
import torch

# Path to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
core_dir = os.path.join(project_root, "core")
src_dir = os.path.join(core_dir, "src")

# Ensure paths are in sys.path
for p in [project_root, core_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.src.Metrics.metrics_function import distance_metrics, return_binary_rain, range_masks, update_dictionary,normalize_dict_by_row
from core.src.Visualization.plot import standardize_tensor_shape

def mask_low_quality_rain(rain, rq, amount=50):
    rain = standardize_tensor_shape(rain, expect_4d=False)
    rq = standardize_tensor_shape(rq, expect_4d=False)
    if rain.shape != rq.shape:
        print("shape rain ", rain.shape, " rq shape ", rq.shape)
        raise ValueError("rain and rq must have the same shape")
    if amount < 0 or amount > 100:
        raise ValueError("amount should be between 0 and 100")
    
    rain = rain.clone().float()
    mask = rq < amount
    rain[mask] = float('nan')
    return rain

def range_metrics(rain, sampled_image, rain_value_threshold=0.1):
    """
    Compute metrics from masked rain and sampled images for different rain intensity ranges.
    
    Args:
        rain: Ground truth rain tensor/array
        sampled_image: Predicted rain tensor/array  
        rain_value_threshold: Threshold for rain detection (default: 0.1)
        
    Returns:
        dict: Dictionary containing:
            - range_metrics: Metrics (RMSE, MAE, CSI, etc.) for each rain category
            - nb_occurence_range: Number of valid pixels for each rain category
            - range_confusions: Confusion matrix counts between rain categories
    """
    
    # Initialize result dictionaries
    range_metrics = {}
    nb_occurence_range = {}
    range_confusions = {}
    
    # Get masks for different rain intensity ranges
    rain_masks, sampled_masks, type_rain = range_masks(rain, sampled_image, rain_value_threshold)
    
    for rain_key in rain_masks.keys():
        # Initialize dictionaries for this rain category
        range_metrics[rain_key] = {}
        nb_occurence_range[rain_key] = 0
        range_confusions[rain_key] = {}
        
        for sampled_key in sampled_masks.keys():
            r_masked = rain_masks[rain_key]  # mask rain image
            s_masked = sampled_masks[sampled_key]  # mask sampled image
            valid = ~torch.isnan(r_masked)
            
            if valid.sum().item() == 0:
                tp = 0
                confusion = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
            else:
                # Compute confusion matrix
                confusion = return_binary_rain(
                    r_masked,
                    s_masked,
                    min_threshold_rain=type_rain[rain_key][0],
                    max_threshold_rain=type_rain[rain_key][1],
                    min_threshold_sampled=type_rain[sampled_key][0],
                    max_threshold_sampled=type_rain[sampled_key][1]
                )
                tp = confusion['tp']
            
            # Update per-range confusion counts
            if sampled_key not in range_confusions[rain_key]:
                range_confusions[rain_key][sampled_key] = 0
            range_confusions[rain_key][sampled_key] += tp
            
            # Only compute metrics when comparing same rain/sampled category
            if rain_key == sampled_key and valid.sum().item() > 0:
                nb_occurence_range[rain_key] += 1
                distance_metrics_result = distance_metrics(
                    rain[valid],
                    sampled_image[valid],
                    rain_value_threshold,
                    metrics_type=['rmse', 'mae', 'csi', 'f1_score', 'accuracy', 'precision', 'recall'],
                    confusion=confusion
                )
                update_dictionary(range_metrics[rain_key], distance_metrics_result)
    
    return {
        'metrics': range_metrics,
        'nb_occurence_range': nb_occurence_range, 
        'confusion': normalize_dict_by_row(range_confusions)
    }

def compute_metrics(rain, sampled, rq):
    # Calculer les masques une seule fois
    rain_masked = mask_low_quality_rain(rain, rq).to(torch.device('cpu'))
    
    sampled_masked = mask_low_quality_rain(sampled, rq).to(torch.device('cpu'))
    
    # Calculer valid mask une fois pour éviter les répétitions
    valid_mask = ~(torch.isnan(rain_masked) | torch.isnan(sampled_masked))
    if valid_mask.sum().item() == 0:
        print("no valid elements")
        return None, None, None  # ou des valeurs par défaut appropriées
    
    # Utiliser les données filtrées
    rain_valid = rain_masked[valid_mask]
    sampled_valid = sampled_masked[valid_mask]
    
    distance_dict = distance_metrics(rain_valid, sampled_valid)
    distance_dict = {k: round(v, 3) if isinstance(v, float) else v for k, v in distance_dict.items()}

    binary_dict = return_binary_rain(rain_valid, sampled_valid)
    range_dict = range_metrics(rain_valid, sampled_valid)

    return distance_dict, binary_dict, range_dict
