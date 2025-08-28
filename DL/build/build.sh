#!/bin/bash

#SBATCH -D /net/nfs/ssd3/cfrancoismartin/

# Job name
#SBATCH --time=3-00:00:00

#SBATCH -J RainDiffusion
#SBATCH --error=/net/nfs/ssd3/cfrancoismartin/Projects/Latmos_Precipitations_Diffusion/DL/experiments/experiment/dump/slurm-%j.err
#SBATCH --output=/net/nfs/ssd3/cfrancoismartin/Projects/Latmos_Precipitations_Diffusion/DL/experiments/experiment/dump/slurm-%j.out
# The number of GPU cards requested.
#SBATCH --gres=gpu:1
# Request 8 CPU cores
#SBATCH --cpus-per-task=5

#SBATCH --mail-user=camille.francois-martin@latmos.ipsl.fr
#SBATCH --mail-type=None

# Overtakes the system memory limits.
ulimit -s unlimited
# Load the user profile
source /etc/profile


# Activate the conda environment
anaconda_bin_path=/net/nfs/ssd3/cfrancoismartin/minforge3/bin
source "${anaconda_bin_path}/activate" myHalEnv
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment"
    exit 1
fi

# Run the Python script
python /net/nfs/ssd3/cfrancoismartin/Projects/Latmos_Precipitations_Diffusion/DL/core/main.py --exp 3 --mode train --config_dir "/net/nfs/ssd3/cfrancoismartin/Projects/Latmos_Precipitations_Diffusion/DL/experiments/best_experiment/configs/e3_config.yaml" \
                                                            --image_dir "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset/2019/sevmos_2019-01-08_09:10:00_4.npy" --debug_run
                                                        

if [ $? -ne 0 ]; then
    echo "Python script failed"
    exit 1
fi
# Print confirmation
echo "Script executed successfully"

# Deactivate conda environment
conda deactivate

# Exit the script
exit 0