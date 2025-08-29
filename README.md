Quantifying Rain Rates from geostationnary satellite Infrared radiometer using a Denoising Diffusion Probabilistic Model


This repository implements and evaluates diffusion models for precipitation prediction, with a focus on deep learning, visualization, and applied case studies.
The organization is designed to clearly separate the model’s core code, experiments, and exploratory analyses.

🌍 Project Overview

This project aims to reconstruct precipitation fields from geostationary satellite infrared (IR) observations using deep generative models.
Unlike low-orbit satellites equipped with microwave sensors that can directly measure rainfall, geostationary satellites only provide IR measurements related to cloud-top temperatures. While this signal is indirect, it contains valuable information about cloud convection, vertical development, and organization, which are strongly correlated with precipitation events.

To capture these complex, nonlinear relationships, we explore the use of deep learning models, with a particular focus on diffusion models (DDPMs). These state-of-the-art generative architectures have shown impressive performance in image modeling and hold promising potential for precipitation estimation and nowcasting.


Code Structure:

Latmos_Precipitations_Diffusion
│
├── DL/                        
│   └── core/                  # Main scripts & core logic
│       ├── eval.py
│       ├── training.py
│       ├── inference.py
│       ├── main.py
│       │
│       └── src/               # Submodules of the core model
│           ├── Dataset/       # Data loading & transformations
│           │   ├── dataset.py
│           │   ├── transform.py
│           │   └── csv/       # Normalization stats (mean/std)
│           │
│           ├── Metrics/       # Evaluation & monitoring metrics
│           │   ├── eval_metrics.py
│           │   ├── metrics_function.py
│           │   └── trainer_metrics.py
│           │
│           ├── Sampling/      # Diffusion process & sampling
│           │   ├── diffusion_constants.py
│           │   └── sample.py
│           │
│           ├── Unet/          # U-Net architecture
│           │   ├── loss.py
│           │   ├── unet.py
│           │   └── Unet_backbone/
│           │       ├── attention.py
│           │       ├── block.py
│           │       ├── sample.py
│           │       └── time_embedding.py
│           │
│           └── Visualization/ # Visualization & plotting
│               ├── eval_plot.py
│               ├── plot.py
│               └── trainer_plot.py
│
├── build/                     
│   └── build.sh               # Build / run automation script
│
├── experiments/               # Saved experiments
│   ├── best_experiment/       # Reference / best experiment
│   │   ├── checkpoint/
│   │   ├── configs/           # This is where the config files are stored, use them for Template <-------
│   │   └── results/
│   │
│   └── experiment/            # General experiments
│       ├── checkpoints/
│       ├── configs/
│       ├── results/
│       └── dump/
│
├── case_study/                # Specific case studies
│   ├── data/
│   ├── results/
│   └── scripts/
│       ├── core/              # core from DL
│       ├── Matthieu/          # U-net by Matthieu Meignin
│       │   └── unet_Matthieu.py
│       ├── dataviz.py
│       ├── generate_rain.py
│       ├── main.py
│       ├── metrics.py
│       └── tiles.py
│
├── EDA/                       # Exploratory Data Analysis
│   ├── results/
│   └── scripts/
│       ├── Matthieu/
│       │   └── unet_Matthieu.py
│       ├── data_analysis.py
│       ├── data_treatment.py
│       ├── fuse_dataset.py
│       └── nan_distribution.py
│
├── README.md                  # Project presentation
├── Report.pdf                 # Project report
├── requirements.txt  



📡 Data Source

The dataset is built from geostationary satellite (MSG metsat) IR radiometer measurements, which provide continuous, high-frequency coverage of large regions.

Input: Infrared brightness temperatures K (multiple IR channels) referenced as tb in the code.

Target: Rainfall estimates from meteofrance radar mosaic referenced as r in the code.

Challenge: The IR–rainfall relationship is indirect and highly variable, requiring advanced inversion techniques to extract meaningful rainfall patterns.

This combination enables the training of data-driven models that can learn to infer precipitation from satellite imagery, bridging the gap between frequent IR observations and sparse direct rainfall measurements.

🧩 Model Architecture
🔹 General Structure

The model is based on a symmetric U-Net composed of three main parts:

🔽 Encoder (downsampling): progressively reduces spatial resolution while increasing feature depth using convolutions.

⚡ Bottleneck: captures global patterns through ResNet blocks and attention modules.

🔼 Decoder (upsampling): reconstructs the target image, reinjecting details via skip connections.

Each stage is enriched with:

⏳ Temporal position embeddings (sinusoidal): inject noise-level information into the network.

🎯 Attention modules: capture long-range spatial dependencies, crucial for reconstructing large rainfall structures.

🔹 Network Workflow

Encoder → strided convolutions + ResNet blocks + attention → compress input into abstract features.

Bottleneck → 2 ResNet blocks + 1 attention block → capture complex global structures.

Decoder → upsampling + ResNet blocks + attention → reconstruct output while leveraging skip connections for fine details.

Final layer → predicts noise, which is subtracted at each diffusion step to generate the denoised image.

🔹 Model Size

The U-Net totals ≈ 9.9M parameters (~39.6 MB).

Component	Parameters
* Initial convolution	640
* Temporal MLP embeddings	82.4k
* Encoder blocks	1.6M
* Bottleneck – ResNet (×2)	1.3M each
* Bottleneck – Attention	131k
* Decoder blocks	5.3M
* Final ResNet block	152k
* Output convolution	65

## 📖 User Guide

### 1. Installation

* Create a virtual environment (recommended).
* Install all required Python packages:

pip install -r requirements.txt

### 2. Prepare the Dataset

* Dataset must contain `.npy` files:

  * **`rain_rate.npy`** → target variable (precipitation rate).
  * **Condition file(s)** → input data (can be named freely).
  * **`rain_quality.npy`** → radar quality indicator (*optional*: not required for core functionality, but needed in some runs).

* Create CSV files for dataset splits (train/val/test).

  * Example scripts for generating splits are provided in the **`EDA/`** folder.

* Ensure the dataset contains **no NaN values**.

### 3. Configuration

* Define a configuration file in **YAML format** containing all training parameters.
* A **template config file** is available in the codebase for reference.

### 4. Build Script

* Create a build script (`.sh`) to launch training or evaluation.
* A **template build file** is included, adapted for the **hal.ipsl.fr** server.

### 5. Running the Code

The entry point is **`main.py`**, which supports three modes:

* 🏋️ **Training**:

  python DL/core/main.py --exp <exp_id> --mode train --config_dir <path_to_config>


* 📊 **Evaluation**:

  python DL/core/main.py --exp <exp_id> --mode eval --config_dir <path_to_config>


* 🔮 **Inference (on a single image)**:

  python DL/core/main.py --exp <exp_id> --mode inference --config_dir <path_to_config> --image_dir <path_to_image>



### 🔑 Arguments

* `--exp` → experiment number (used to organize results in `experiments/`).
* `--mode` → one of: `train`, `eval`, `inference`.
* `--config_dir` → path to the YAML config file.
* `--image_dir` → path to the input image (**only required for inference mode**).
* `--debug_run` → add it for using the code on a small subset of the dataset

👥 Contributors : Camille Francois Martin master student at Latmos, Meignin Matthieu Phd student at Latmos
