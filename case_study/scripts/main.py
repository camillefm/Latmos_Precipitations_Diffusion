import yaml
import torch
import numpy as np
#9 aout 2019
import sys
import os

projects_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
rain_diffusion_dir = os.path.join(projects_dir, "RainDiffusion")

if projects_dir not in sys.path:
    sys.path.insert(0, projects_dir)

if rain_diffusion_dir not in sys.path:
    sys.path.insert(0, rain_diffusion_dir)

from RainDiffusion.Inference.inference import load_image, load_model, inference, save_plot
from data_viz import extraction, radar_quality, log_metrics, plot_confusion_matrix, plot_histogram_distribution, plot_range_metrics, plot_difference, plot_differences_only, plot_density_heatmap
from generate_rain import generate_sampled_rain, generate_cnn_rain,generate_cnn_rain_tiled,generate_rain_mean
from metrics import compute_metrics

config_dir = "/net/nfs/ssd3/cfrancoismartin/Projects/RainDiffusion/_Runs/configs/e3_config.yaml"
result_dir = "/net/nfs/ssd3/cfrancoismartin/Projects/ClimMatthieu/results"
data_dir = "/net/nfs/ssd3/cfrancoismartin/Projects/ClimMatthieu/data/sevmos_2019-08-09_18:10:00.nc"
image_name = "e3_simple"
load_npy_image = True
result_dir += "/" + image_name
def load_config(path):
    """
    Load the configuration file from the given path.
    """
    print(f"📄 Loading configuration from: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config(config_dir)
image_size = config['model']['image_size'] #square image
#loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda available:", torch.cuda.is_available())
model = load_model(config, device)


# Extract and process data
data = extraction(data_dir)
data = radar_quality(data)

if not load_npy_image:
    print("generation image ... ", flush = True)
#generate image Diffusion
    
    #tb_full_cropped, r_full_cropped, sampled_full, rq_full_cropped = generate_sampled_rain(model, config, data, device=device)
    tb_full_cropped, r_full_cropped, sampled_full, rq_full_cropped = generate_rain_mean(model, config, data, number_of_gen=10, device=device, only_crop = False)
    sampled_cnn_full = generate_cnn_rain(tb_full_cropped, device)
    
    #sampled_cnn_full = generate_cnn_rain_tiled(tb_full_cropped, image_size, device, model=None)

    # Convert tensor to numpy array
    diffusion_np = sampled_full.cpu().numpy()
    cnn_np = sampled_cnn_full.cpu().numpy()  # Use .cpu() if tensor is on GPU

    # Save numpy array to .npy file
    np.save(result_dir +"/"+image_name+ "_cnn_tensor.npy" , cnn_np)
    np.save(result_dir +"/"+image_name+ "_diffusion_tensor.npy", diffusion_np)
else :
    
    tb_full_cropped, r_full_cropped, _, rq_full_cropped = generate_sampled_rain(model, config, data, device=device, only_crop=True)
    
    print("Loading the image from npy files ... ", flush = True)
    # Load the numpy array
    cnn_np = np.load(result_dir + "/"+ image_name + "_cnn_tensor.npy")
    diffusion_np = np.load(result_dir +"/"+image_name+ "_diffusion_tensor.npy")
    # Convert numpy array to tensor
    sampled_cnn_full = torch.from_numpy(cnn_np)
    sampled_full = torch.from_numpy(diffusion_np)

tb_plot = tb_full_cropped.unsqueeze(0).repeat(2, 1, 1, 1).to(torch.device('cpu'))     # (2,C, H, W)
r_plot = r_full_cropped.repeat(2, 1, 1).to(torch.device('cpu'))       # (2, H, W)

sampled_plot = torch.cat([sampled_full, sampled_cnn_full.squeeze(0)], dim=0).to(torch.device('cpu'))  # different tensors
rq_plot = rq_full_cropped.repeat(2, 1, 1).to(torch.device('cpu'))       # (2, H, W)

save_plot(tb_plot , r_plot , sampled_plot, rq_plot, result_dir, image_name +"full_image" ,config, normalize_tb_nb_of_sigmas=0)

distance_dict_diffusion, binary_dict_diffusion, range_dict_diffusion = compute_metrics(r_full_cropped, sampled_full, rq_full_cropped)
distance_dict_cnn, binary_dict_cnn, range_dict_cnn = compute_metrics(r_full_cropped, sampled_cnn_full, rq_full_cropped)

log_metrics(distance_dict_diffusion, png_path=result_dir+"/" + image_name +"_distance_metrics.png", cnn_metric=distance_dict_cnn)
plot_range_metrics(range_dict_diffusion['metrics'], png_path=result_dir +"/" + image_name + "_range_metrics.png", cnn_metric=range_dict_cnn['metrics'])
plot_histogram_distribution(r_full_cropped, sampled_full, result_dir +"/" + image_name +"_rain_distribution_histogram.png", cnn_sampled=sampled_cnn_full)
plot_confusion_matrix(range_dict_diffusion['confusion'], result_dir  +"/" + image_name +"_rain_confusion_matrix.png", range_dict_cnn['confusion'])
plot_differences_only(diffusion_image=sampled_full.squeeze(), target = r_full_cropped, png_path =  result_dir +"/" + image_name +"_difference_only", cnn_image = sampled_cnn_full.squeeze())
plot_density_heatmap(r_full_cropped, sampled_full, result_dir +"/" + image_name +"_rain_scatter.png", cnn_sampled=sampled_cnn_full)