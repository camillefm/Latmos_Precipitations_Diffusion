#9 aout 2019
import sys
import os

# Path to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
core_dir = os.path.join(project_root, "core")
src_dir = os.path.join(core_dir, "src")

# Ensure paths are in sys.path
for p in [project_root, core_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)
        
from core.inference import load_image, load_model, inference, save_plot
from case_study.scripts.tiles import split_tensor_to_grid,merge_tiles_to_tensor

from case_study.scripts.Matthieu.unet_Matthieu import UnetMatthieu

import torch
import numpy as np
from tqdm import tqdm

def inverse( tensor):
    """
    "Inverse" operation clips negatives back to zero.

    This ensures output stays non-negative after noise injection.
    """
    tensor = tensor.clone()
    tensor[tensor < 0] = 0
    return tensor



def generate_sampled_rain(model, config, data, device=None, only_crop= False):
    """
    Generate full sampled image tensor from data using model and config.

    Parameters:
    - model: PyTorch model
    - config: config dict, with keys 'model'->'image_size' and 'list_channels'
    - data: dict with keys for all channels in list_channels, plus 'rain_rate' and 'rain_quality'
    - device: torch device (default: cuda if available else cpu)

    Returns:
    - sampled_full: tensor with full sampled image (C, H, W)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = config['model']['image_size']
  
        
        

    tb_channels = []
    for channel_name in config['list_channels']:
        if channel_name != 'cnn_output':
            arr = data[channel_name]
            tb_channels.append(torch.tensor(arr).to(device))
    
    tb_full = torch.stack(tb_channels).to(device)
    if 'cnn_output' in config['list_channels']:
        cnn_output = generate_cnn_rain(tb_full,device=device).squeeze()
        tb_channels.append(torch.tensor(cnn_output))
        tb_full = torch.stack(tb_channels).to(device)

    r_full = torch.tensor(data['rain_rate'], device=device).unsqueeze(0)
    rq_full = torch.tensor(data['rain_quality'], device=device).unsqueeze(0)

    tb_grid, tb_grid_shape = split_tensor_to_grid(tb_full, image_size, image_size)
    r_grid, r_grid_shape = split_tensor_to_grid(r_full, image_size, image_size)
    rq_grid, rq_grid_shape = split_tensor_to_grid(rq_full, image_size, image_size)

    assert tb_grid_shape == r_grid_shape == rq_grid_shape, "Grid shapes don't match"

    if not only_crop:
        sampled_grid = []
        for i in tqdm(range(len(tb_grid)), desc="Running inference on tiles"):
            tb_i, r_i, rq_i = tb_grid[i], r_grid[i], rq_grid[i]
            _, _, sampled_i, _ = inference(model, tb_i, r_i, rq_i, config, device)
            sampled_grid.append(sampled_i)
            #to get normalized tb for plot

        sampled_full = merge_tiles_to_tensor(sampled_grid, tb_grid_shape, image_size, image_size)
    else:
        sampled_full = None
    r_full_cropped = merge_tiles_to_tensor(r_grid, tb_grid_shape, image_size, image_size)
    tb_full_cropped = merge_tiles_to_tensor(tb_grid, tb_grid_shape, image_size, image_size)
    rq_full_cropped = merge_tiles_to_tensor(rq_grid, tb_grid_shape, image_size, image_size)

    return tb_full_cropped, r_full_cropped, sampled_full, rq_full_cropped

def generate_cnn_rain(tb, device, model=None):
    """
    Generate CNN rain prediction from brightness temperature data.
    
    Args:
        tb: Tensor of shape (C, H, W) with all 7 channels in correct order
        device: Target device for computation
        model: Pre-loaded model (optional, for efficiency)
    
    Returns:
        Rain prediction tensor
    """
    input_list = ['IR_087','IR_108','IR_120','WV_062','WV_073','IR_134','IR_097']
    channel_stats = {
        'IR_087': (255.37857118110065, 17.261302580374732),
        'IR_097': (238.69379756036795, 9.316924399682433),
        'IR_108': (256.45468038526633, 18.288274434661673),
        'IR_120': (255.24329992505753, 18.151034453886655),
        'IR_134': (243.01295702873065, 11.53267337341514),
        'WV_062': (230.50257753091145, 5.454924889447568),
        'WV_073': (242.24005631850198, 9.69377899523295)
    }
    

    
    working_script_order = ['IR_087','IR_108','IR_120','WV_062','WV_073','IR_134','IR_097']
    
    # Build mapping assuming tb follows working_script_order
    channel_to_index = {name: idx for idx, name in enumerate(working_script_order)}
    
    # Extract channels in the order expected by the model (input_list order)
    selected_indices = [channel_to_index[name] for name in input_list]
    tb_selected = tb[selected_indices]  # Shape: (7, H, W)
    
    # Move to CPU for normalization (ensuring float32)
    tb_selected_cpu = tb_selected.detach().cpu().numpy().astype(np.float32)
    
    # Normalize using the same method as working script
    normalized_tb = np.empty_like(tb_selected_cpu)
    for i, name in enumerate(input_list):
        mean, std = channel_stats[name]
        normalized_tb[i] = (tb_selected_cpu[i] - mean) / (3 * std)
    
    # Convert to tensor with correct device and dtype
    tensor_input = torch.tensor(normalized_tb, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Load model if not provided
    if model is None:
        weight_path = "/net/nfs/ssd3/mmeignin/RainSat/DL/regression/experiments/experiment41/checkpoints/best_model.pth"
        unet_matthieu = UnetMatthieu(
            n_channels=len(input_list),
            n_classes=1,
            features=[32, 64, 128, 256, 512],
            bilinear=False,
            dropout=0.05
        )
        
        print(f"Loading CNN weights from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        unet_matthieu.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")
        unet_matthieu.to(device)
        unet_matthieu.eval()
        model = unet_matthieu
    
    # Generate prediction
    with torch.no_grad():
        output = inverse(model(tensor_input))
    
    return output

def generate_cnn_rain_tiled(tb_full, image_size, device, model=None):
    """
    Generate CNN rain prediction from brightness temperature data using tiling approach.
    
    Args:
        tb_full: Tensor of shape (C, H, W) with all 7 channels in working_script_order
        image_size: Size of tiles for processing (e.g., 256)
        device: Target device for computation
        model: Pre-loaded CNN model (optional, for efficiency)
    
    Returns:
        rain_prediction_full: Full rain prediction tensor reassembled from tiles
    """
    from tqdm import tqdm
    import torch
    import numpy as np
    
    # Split the full tensor into tiles
    tb_grid, tb_grid_shape = split_tensor_to_grid(tb_full, image_size, image_size)
    
    # Load model if not provided (same as in generate_cnn_rain)
    if model is None:
        weight_path = "/net/nfs/ssd3/mmeignin/RainSat/DL/regression/experiments/experiment41/checkpoints/best_model.pth"
        unet_matthieu = UnetMatthieu(
            n_channels=7,  # 7 channels for the CNN
            n_classes=1,
            features=[32, 64, 128, 256, 512],
            bilinear=False,
            dropout=0.05
        )
        
        print(f"Loading CNN weights from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        unet_matthieu.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")
        unet_matthieu.to(device)
        unet_matthieu.eval()
        model = unet_matthieu
    
    # Process each tile
    rain_prediction_grid = []
    for i in tqdm(range(len(tb_grid)), desc="Running CNN inference on tiles"):
        tb_tile = tb_grid[i]  # Shape: (C, tile_H, tile_W)
        
        # Generate rain prediction for this tile using the existing function logic
        rain_prediction_tile = generate_cnn_rain(tb_tile, device, model)
        
        # Remove batch dimension if present and ensure correct shape
        if rain_prediction_tile.dim() == 4:  # (1, 1, H, W)
            rain_prediction_tile = rain_prediction_tile.squeeze(0)  # (1, H, W)
        
        rain_prediction_grid.append(rain_prediction_tile)
    
    # Merge tiles back to full tensor
    rain_prediction_full = merge_tiles_to_tensor(
        rain_prediction_grid, 
        tb_grid_shape, 
        image_size, 
        image_size
    )
    
    return rain_prediction_full


def generate_rain_mean(model, config, data, number_of_gen, device, only_crop = False):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image_size = config['model']['image_size']
    
            
            

        tb_channels = []
        for channel_name in config['list_channels']:
            if channel_name != 'cnn_output':
                arr = data[channel_name]
                tb_channels.append(torch.tensor(arr).to(device))
        
        tb_full = torch.stack(tb_channels).to(device)
        if 'cnn_output' in config['list_channels']:
            cnn_output = generate_cnn_rain(tb_full,device=device).squeeze()
            tb_channels.append(torch.tensor(cnn_output))
            tb_full = torch.stack(tb_channels).to(device)

        r_full = torch.tensor(data['rain_rate'], device=device).unsqueeze(0)
        rq_full = torch.tensor(data['rain_quality'], device=device).unsqueeze(0)

        tb_grid, tb_grid_shape = split_tensor_to_grid(tb_full, image_size, image_size)
        r_grid, r_grid_shape = split_tensor_to_grid(r_full, image_size, image_size)
        rq_grid, rq_grid_shape = split_tensor_to_grid(rq_full, image_size, image_size)

        assert tb_grid_shape == r_grid_shape == rq_grid_shape, "Grid shapes don't match"

        
        if not only_crop:
            sampled_list = []
            for _ in range(number_of_gen):
                sampled_grid = []
                for i in tqdm(range(len(tb_grid)), desc="Running inference on tiles"):
                    tb_i, r_i, rq_i = tb_grid[i], r_grid[i], rq_grid[i]
                    _, _, sampled_i, _ = inference(model, tb_i, r_i, rq_i, config, device)
                    sampled_grid.append(sampled_i)
                    #to get normalized tb for plot

                sampled_full = merge_tiles_to_tensor(sampled_grid, tb_grid_shape, image_size, image_size)
                sampled_list.append(sampled_full)

        sampled_mean = torch.stack(sampled_list, dim=0).mean(dim=0)
        
        r_full_cropped = merge_tiles_to_tensor(r_grid, tb_grid_shape, image_size, image_size)
        tb_full_cropped = merge_tiles_to_tensor(tb_grid, tb_grid_shape, image_size, image_size)
        rq_full_cropped = merge_tiles_to_tensor(rq_grid, tb_grid_shape, image_size, image_size)

        return tb_full_cropped, r_full_cropped, sampled_mean, rq_full_cropped