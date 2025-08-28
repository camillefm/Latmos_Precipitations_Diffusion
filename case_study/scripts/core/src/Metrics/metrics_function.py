import torch
import numpy as np
from scipy.stats import skew, kurtosis


# Safe division helper
def safe_div(n, d): 
    return n / d if d != 0 else 0

def distance_metrics(rain, sampled_image, rain_value_threshold=0.1,
                     metrics_type=['rmse', 'mae', 'precision', 'recall', 'f1_score', 'accuracy', 'csi'], confusion = None):
    
    # Ensure tensors are torch.Tensor
    if isinstance(sampled_image, np.ndarray):
        sampled_image = torch.tensor(sampled_image)
    if isinstance(rain, np.ndarray):
        rain = torch.tensor(rain)

    rain = rain.to(sampled_image.device).flatten()
    sampled_image = sampled_image.flatten()

    # Filter out NaNs from both rain and sampled_image
    valid = ~torch.isnan(rain) & ~torch.isnan(sampled_image)
    rain = rain[valid]
    sampled_image = sampled_image[valid]

    # Avoid empty tensors
    if len(rain) == 0:
        rain = torch.tensor([0.0], device=sampled_image.device)
        sampled_image = torch.tensor([0.0], device=sampled_image.device)

    metrics = {}

    # RMSE and MAE
    diff = sampled_image - rain
    if 'rmse' in metrics_type:
        metrics['rmse'] = torch.sqrt(torch.nanmean(diff ** 2)).item()
    if 'mae' in metrics_type:
        metrics['mae'] = torch.nanmean(torch.abs(diff)).item()

    # Classification metrics
    if any(m in metrics_type for m in ['precision', 'recall', 'f1_score', 'accuracy', 'csi']):
        if confusion is None:
            confusion = return_binary_rain(rain, sampled_image,
                                       min_threshold_rain=rain_value_threshold,
                                       min_threshold_sampled=rain_value_threshold)
        tp = confusion["tp"]
        fp = confusion["fp"]
        fn = confusion["fn"]
        tn = confusion["tn"]

        if 'precision' in metrics_type:
            metrics['precision'] = safe_div(tp, tp + fp)
        if 'recall' in metrics_type:
            metrics['recall'] = safe_div(tp, tp + fn)
        if 'f1_score' in metrics_type:
            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            metrics['f1_score'] = safe_div(2 * p * r, p + r)
        if 'accuracy' in metrics_type:
            metrics['accuracy'] = safe_div(tp + tn, tp + fp + fn + tn)
        if 'csi' in metrics_type:
            metrics['csi'] = safe_div(tp, tp + fp + fn)

    #early stopping metrics 
    if 'rmse' in metrics_type and 'csi' in metrics_type:
        metrics['early stopping rmse*(1-csi)'] = metrics['rmse'] * (1-metrics['csi'])
    return metrics



def anormality_metrics(sampled_image):

    if isinstance(sampled_image, np.ndarray):
        sampled_image = torch.tensor(sampled_image)

    nb_of_extreme_high_pixels = (sampled_image > 300.0).sum().item()
    nb_of_subzero_pixels = (sampled_image < 0.0).sum().item()

    return {
        'nb_of_extreme_high_pixels': nb_of_extreme_high_pixels,
        'nb_of_subzero_pixels': nb_of_subzero_pixels
        }


def replace_image(tuple_image, criterion, list_image, list_criterion, criterion_name='rmse', order='best'):
    """
    Replace the image in list_image only if the new criterion is better (or worse) based on the order.
    """

    if isinstance(criterion, torch.Tensor):
        criterion = criterion.item()

    if order == "best":
        if criterion_name in ['rmse', 'mae']:
            # Lower is better
            worst_index = list_criterion.index(max(list_criterion))
            if criterion < list_criterion[worst_index]:
                list_criterion[worst_index] = criterion
                list_image[worst_index] = tuple_image

        elif criterion_name in ['csi', 'f1_score', 'accuracy', 'precision', 'recall']:
            # Higher is better
            worst_index = list_criterion.index(min(list_criterion))
            if criterion > list_criterion[worst_index]:
                list_criterion[worst_index] = criterion
                list_image[worst_index] = tuple_image
        else:
            raise ValueError(f"Unknown criterion name: {criterion_name}.")

    elif order == "worst":
        if criterion_name in ['rmse', 'mae']:
            # Higher is worse
            best_index = list_criterion.index(min(list_criterion))
            if criterion > list_criterion[best_index]:
                list_criterion[best_index] = criterion
                list_image[best_index] = tuple_image

        elif criterion_name in ['csi', 'f1_score', 'accuracy', 'precision', 'recall']:
            # Lower is worse
            best_index = list_criterion.index(max(list_criterion))
            if criterion < list_criterion[best_index]:
                list_criterion[best_index] = criterion
                list_image[best_index] = tuple_image
        else:
            raise ValueError(f"Unknown criterion name: {criterion_name}.")

    elif order == "random":
        index = np.random.randint(0, len(list_image))
        list_image[index] = tuple_image

    else:
        raise ValueError(f"Unknown order: {order}. Use 'best', 'worst', or 'random'.")

    return list_image, list_criterion



def pixel_metrics(rain_pixel_values, sampled_pixel_values):
    skew_rain = skew(rain_pixel_values)
    skew_sampled = skew(sampled_pixel_values)
    kurtosis_rain = kurtosis(rain_pixel_values)
    kurtosis_sampled = kurtosis(sampled_pixel_values)

    metrics = {
        'skew_rain': skew_rain,
        'skew_sampled': skew_sampled,
        'kurtosis_rain': kurtosis_rain,
        'kurtosis_sampled': kurtosis_sampled
    }
    return metrics

def range_masks(rain, sampled_image, rain_value_threshold=0.1):
    type_rain = {
        'no_rain': [-100, rain_value_threshold],
        'slight_rain': [rain_value_threshold, 2.5],
        'moderate_rain': [2.5, 10],
        'heavy_rain': [10, 50],
        'violent_rain': [50, 350]
    }

    # Replace values outside the range with NaN
    def apply_nan_mask(data, low, high):
        return torch.where((data >= low) & (data < high), data, torch.nan)

    # Create masked tensors with NaN for both rain and sampled_image
    rain_nan_masks = {
        key: apply_nan_mask(rain, value[0], value[1])
        for key, value in type_rain.items()
    }

    sampled_nan_masks = {
        key: apply_nan_mask(sampled_image, value[0], value[1])
        for key, value in type_rain.items()
    }

    return rain_nan_masks, sampled_nan_masks, type_rain


def return_binary_rain(rain, sampled_image, min_threshold_rain=0.1, max_threshold_rain=350, min_threshold_sampled=0.1, max_threshold_sampled=350):
    """
    Compute the confusion matrix for the given rain and sampled images.
    """
    pred_mask = (sampled_image >= min_threshold_sampled) & (sampled_image <= max_threshold_sampled)
    true_mask = (rain >= min_threshold_rain) & (rain <= max_threshold_rain)

    tp = torch.sum(pred_mask & true_mask).item()
    fp = torch.sum(pred_mask & ~true_mask).item()
    fn = torch.sum(~pred_mask & true_mask).item()
    tn = torch.sum(~pred_mask & ~true_mask).item()

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

def update_dictionary(target_dictionary, dictionary):
    """
    Update a dictionary with a key-value pair.
    If the key already exists, add the value to the existing value.
    """
    for key, value in dictionary.items():
        if key not in target_dictionary:
            target_dictionary[key] = 0
        target_dictionary[key] += value
    return target_dictionary

def normalize_dictionary(dictionary, nb_samples):
    """
    Normalize the values in a dictionary by the number of samples.
    """
    for key in dictionary:
        if nb_samples == 0:
            dictionary[key] = 0
        else:
            dictionary[key] /= nb_samples


def normalize_dict_by_row(data):
    """
    Normalize a dictionary of dictionaries by row.
    Each inner dictionary is treated as a row, and its values are normalized
    so that the sum of values equals 1.
    
    Args:
        data (dict): Dictionary of dictionaries to normalize.
        
    Returns:
        dict: Normalized dictionary.
    """
    normalized = {}
    for outer_key, inner_dict in data.items():
        total = sum(inner_dict.values())
        if total == 0:
            # Avoid division by zero
            normalized[outer_key] = {k: 0 for k in inner_dict}
        else:
            normalized[outer_key] = {k: v / total for k, v in inner_dict.items()}
    return normalized
