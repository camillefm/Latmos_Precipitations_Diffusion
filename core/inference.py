import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
import ast 

from src.Unet.unet import Unet
from src.Visualization.plot import plot_tensors
from src.Sampling.sample import sample
from src.Dataset.transforms import SharedRandomCrop3D_XYRQ


def load_model(config, device):
    checkpoint_name = config['training']['checkpoint_name']
    checkpoint_dir = config['checkpoint_directory']
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f"Loading model from checkpoint {checkpoint_path} ...")
    
    model = Unet.load_from_checkpoint(checkpoint_path, config=config, device=device)
    model.eval()
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    trained_epoch = checkpoint['epoch']
    del checkpoint
    
    print(f"Model loaded at epoch {trained_epoch}")
    return model

def load_image(image_path, input_list, device):
    data = np.load(image_path, allow_pickle=True).item()
    tb = np.stack([data[IR_name] for IR_name in input_list]).astype(np.float32)  # (C, H, W)
    r = np.array(data['rain_rate'], dtype=np.float32)
    rq = np.array(data['rain_quality'], dtype=np.float32)
    
    tb_tensor = torch.tensor(tb, device=device)
    r_tensor = torch.tensor(r, device=device).unsqueeze(0)
    rq_tensor = torch.tensor(rq, device=device).unsqueeze(0)
    return tb_tensor, r_tensor, rq_tensor

def inference(model, tb, r, rq, config, device):
    random_cropper = SharedRandomCrop3D_XYRQ(crop_size=config['model']['image_size'])
    
    if config['transforms']['random_crop']:
        tb, r, rq = random_cropper(tb, r, rq)
    
    tb = tb.unsqueeze(0)  # batch dim
    r = r.unsqueeze(0)
    rq = rq.unsqueeze(0)
    
    if config['transforms']['normalize_tb'] == "mean_std":
        tb_stats = pd.read_csv(config['norm_csv_directory'] + f"e{config['num_experiment']}_csv_tb.csv")
        mean_list = ast.literal_eval(tb_stats['mean'].iloc[0])[0]
        std_list = ast.literal_eval(tb_stats['std'].iloc[0])[0]
        mean_tb = torch.tensor(mean_list, dtype=torch.float32).view(1, -1, 1, 1).to(device)
        std_tb = torch.tensor(std_list, dtype=torch.float32).view(1, -1, 1, 1).to(device)
        tb = (tb - mean_tb) / (std_tb * config['transforms']['normalize_tb_nb_of_sigmas'])
    
    sampled_image = sample(
        model,
        tb,
        timesteps=config['diffusion']['timesteps'],
        normalized=config['transforms']['normalize_rain'] == "mean_std",
        log1p=config['transforms']['rain_log_normalization'],
        init_noise=None,
        return_all_steps=False,
        nb_sigmas_rain=config['transforms']['normalize_rain_nb_of_sigmas']
    ).to(device)
    
    
    return tb, r, sampled_image, rq


def save_plot(tb_tensor, rain_tensor, sampled_tensor, rq_tensor, output_dir, image_name, config, normalize_tb_nb_of_sigmas = None):
    sampled_tensor = torch.where(sampled_tensor > 0.1, sampled_tensor, torch.nan)
    if normalize_tb_nb_of_sigmas is None:
        normalize_tb_nb_of_sigmas = config['transforms']['normalize_tb_nb_of_sigmas']
    output_image = plot_tensors(
        tb_tensor=tb_tensor,
        rain_tensor=rain_tensor,
        sampled_tensor=sampled_tensor,
        rq_tensor=rq_tensor,
        tb_normalised_nb_sigmas=normalize_tb_nb_of_sigmas
    )

    if not image_name.lower().endswith(".png"):
        image_name += ".png"
    
    output_path = os.path.join(output_dir,  f"output_e{config['num_experiment']}_{image_name}.png")

    img = Image.fromarray(output_image)
    img.save(output_path)
    print(f"Image saved to {output_path}")

# Usage example inside your script:
def inference_ddpm_model(config, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda available:", torch.cuda.is_available())

    model = load_model(config, device)
    tb, r, rq = load_image(image_path, config["list_channels"], device)
    tb, r, sampled_image, rq = inference(model, tb, r, rq, config, device)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    

    save_plot(tb, r, sampled_image, rq, output_dir=config['result_directory'], image_name = filename, config =config)
    print("Inference done!")
