
import torch
import yaml
import argparse

from Training.training import train_ddpm_model
from Evaluation.eval import eval_ddpm_model
from Inference.inference import inference_ddpm_model
import time


start = time.time()

def load_config(path):
    """
    Load the configuration file from the given path.
    """
    print(f"📄 Loading configuration from: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser(description="Run DDPM experiment")
parser.add_argument("--exp", type=int, default=1, help="Experiment number")
parser.add_argument("--mode", type=str, default='train', help="Mode (train ,eval, test)")
parser.add_argument("--debug_run", action='store_true', help="Enable debug mode for a smaller run")
parser.add_argument("--config_dir", type=str, required=True, help = "path of the config file")
parser.add_argument("--image_dir", type =str ,default = None, required=False, help = "image path for inference")
args = parser.parse_args()
num_experiment = args.exp
debug_run = args.debug_run
mode = args.mode
data_dir_config = args.config_dir
image_path = args.image_dir


# Load configuration from YAML file
config = load_config(data_dir_config)
# Default experiment number
assert num_experiment == config['num_experiment'], f"Experiment number in config ({config['num_experiment']}) does not match the expected number ({num_experiment}). Please check the configuration file."

print(f"Running experiment number: {num_experiment}", flush=True)

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
torch.set_float32_matmul_precision('medium')
# Start training the DDPM model
if mode == 'train':
    print("🚀 Starting training...", flush=True)
    train_ddpm_model(config, debug_run=debug_run)

elif mode == 'eval':
    
    print("🚀 Starting evaluation...", flush=True)
    eval_ddpm_model(config, debug_run=debug_run, checkpoint_name=config['training']['checkpoint_name'])

elif mode == 'inference':
    print("🚀 Starting inference...", flush=True)
    inference_ddpm_model(config, image_path = image_path)



end = time.time()
print(f"Script executed in {end - start:.4f} seconds", flush=True)
print("✅ Experiment completed successfully!", flush=True)