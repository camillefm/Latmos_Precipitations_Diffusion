# 🌧️ Quantifying Rain Rates from Geostationary Satellites with Diffusion Models

This repository implements and evaluates **Denoising Diffusion Probabilistic Models (DDPMs)** for precipitation prediction, with a focus on **deep learning, visualization, and applied case studies**.
The organization is designed to clearly separate the **core model code**, **experiments**, and **exploratory analyses**.

---

## 🌍 Project Overview

Geostationary satellites (e.g., MSG) provide continuous **infrared (IR) radiometer measurements**, which indirectly reflect cloud-top temperatures.
Unlike **low-orbit satellites** with microwave sensors that can directly measure rainfall, geostationary satellites require **inversion techniques** to estimate precipitation.

This project explores **diffusion models (DDPMs)** as a deep generative approach to reconstruct precipitation fields from IR data.

* **Input:** IR brightness temperatures (`tb`) from multiple channels.
* **Target:** Rainfall estimates from Météo-France radar mosaic (`r`).
* **Challenge:** The IR–rainfall relationship is **indirect and highly nonlinear**, demanding advanced modeling.
* **Solution:** Use **U-Net + diffusion models** to bridge the gap between frequent IR observations and sparse rainfall measurements.

---

## 📂 Code Structure

```bash
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
│           ├── Dataset/       # Data loading & preprocessing
│           ├── Metrics/       # Evaluation & monitoring
│           ├── Sampling/      # Diffusion process & sampling
│           ├── Unet/          # U-Net architecture & backbone
│           └── Visualization/ # Plots & visualizations
│
├── build/                     # Build & run scripts
├── experiments/               # Checkpoints, configs, results
├── case_study/                # Applied case studies
├── EDA/                       # Exploratory data analysis
├── README.md                  # Project presentation
├── Report.pdf                 # Detailed report
└── requirements.txt           # Dependencies
```

---

## 🧩 Model Architecture

### 🔹 General Structure

The network is a **symmetric U-Net** with:

* 🔽 **Encoder:** downsampling with strided convolutions + ResNet blocks + attention.
* ⚡ **Bottleneck:** 2 ResNet blocks + 1 attention block for global context.
* 🔼 **Decoder:** upsampling with skip connections + ResNet blocks + attention.
* 🎯 **Final layer:** predicts noise to denoise at each diffusion step.

Each stage uses:

* ⏳ **Temporal embeddings (sinusoidal)** to encode noise level.
* 🎯 **Attention modules** for long-range dependencies.

### 🔹 Model Size

| Component                | Parameters            |
| ------------------------ | --------------------- |
| Initial convolution      | 640                   |
| Temporal MLP embeddings  | 82.4k                 |
| Encoder blocks           | 1.6M                  |
| Bottleneck – ResNet (×2) | 1.3M each             |
| Bottleneck – Attention   | 131k                  |
| Decoder blocks           | 5.3M                  |
| Final ResNet block       | 152k                  |
| Output convolution       | 65                    |
| **Total**                | **≈9.9M (\~39.6 MB)** |

---

## 📖 User Guide

### 1️⃣ Installation

```bash
# Recommended: create a virtual environment
pip install -r requirements.txt
```

---

### 2️⃣ Prepare the Dataset

Your dataset must include `.npy` files:

* **`rain_rate.npy`** → precipitation target.
* **Condition file(s)** → input data (free naming).
* **`rain_quality.npy`** → radar quality flag (*optional*).

👉 Create CSV files for **train/val/test splits** (see **`EDA/`** scripts).
👉 Ensure **no NaN values** in the dataset.

---

### 3️⃣ Configuration

* Use a `.yaml` config file with all training parameters.
* Templates are available in **`experiments/best_experiment/configs/`**.

---

### 4️⃣ Build Script

* Use a `.sh` script to launch runs.
* A template is included for **hal.ipsl.fr**.

---

### 5️⃣ Running the Code

Entry point: **`main.py`**

#### 🏋️ Train

```bash
python DL/core/main.py --exp 1 --mode train --config_dir configs/train.yaml
```

#### 📊 Evaluate

```bash
python DL/core/main.py --exp 1 --mode eval --config_dir configs/train.yaml
```

#### 🔮 Inference

```bash
python DL/core/main.py --exp 1 --mode inference --config_dir configs/train.yaml --image_dir path/to/image.npy
```

---

### 🔑 Arguments

| Argument       | Description                                         |
| -------------- | --------------------------------------------------- |
| `--exp`        | Experiment ID (saves results under `experiments/`). |
| `--mode`       | `train`, `eval`, or `inference`.                    |
| `--config_dir` | Path to YAML config file.                           |
| `--image_dir`  | Path to input image (only for inference).           |
| `--debug_run`  | Run on a small subset (debug mode).                 |

---

## 📡 Data Source

* **Input:** MSG satellite IR radiometer (brightness temperatures in Kelvin).
* **Target:** Météo-France radar rainfall mosaic.
* **Challenge:** IR-to-rainfall mapping is indirect & nonlinear.
* **Goal:** Learn robust inversion with **diffusion models**.

---

## 👥 Contributors

* **Camille François Martin** – Master student @ LATMOS
* **Matthieu Meignin** – PhD student @ LATMOS
