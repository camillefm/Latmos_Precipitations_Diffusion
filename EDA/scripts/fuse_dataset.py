import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from Matthieu.unet_Matthieu import UnetMatthieu
import torch
from matplotlib.colors import LogNorm
import time

def inverse( tensor):
    """
    "Inverse" operation clips negatives back to zero.

    This ensures output stays non-negative after noise injection.
    """
    tensor = tensor.clone()
    tensor[tensor < 0] = 0
    return tensor


file_dir = "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/third_dataset/dataset"
# year = "2008"
# file_name = "sevmos_2008-01-02_12:40:00_2.npy"
# file_path = file_dir + "/" + year + "/" + file_name
input_list = ['IR_087','IR_108','IR_120','WV_062','WV_073','IR_134','IR_097'] 
years = [ "2017", "2018", "2019", "2020", "2021", "2022", "2023"]
channel_stats = {
    'IR_087': (255.37857118110065, 17.261302580374732),
    'IR_097': (238.69379756036795, 9.316924399682433),
    'IR_108': (256.45468038526633, 18.288274434661673),
    'IR_120': (255.24329992505753, 18.151034453886655),
    'IR_134': (243.01295702873065, 11.53267337341514),
    'WV_062': (230.50257753091145, 5.454924889447568),
    'WV_073': (242.24005631850198, 9.69377899523295)
}


weight_path =  "/net/nfs/ssd3/mmeignin/RainSat/DL/regression/experiments/experiment41/checkpoints/best_model.pth"
unet_matthieu = UnetMatthieu(n_channels = len(input_list), n_classes = 1, features=[32, 64, 128, 256,512], bilinear=False, dropout=0.05)
# Load the pre-trained weights for the CNN
print(f"Loading CNN weights from {weight_path}")
    # Load checkpoint
checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
unet_matthieu.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")
print("CNN weights loaded successfully.")
unet_matthieu.eval()
for year in years:
    print(f"Processing year: {year}",flush=True)
    start = time.time()
    for file_name in os.listdir(os.path.join(file_dir, year)):
        file_path = os.path.join(file_dir, year, file_name)
        data = np.load(file_path, allow_pickle=True).item()

        tb = np.stack([data[IR_name] for IR_name in input_list]).astype(np.float32)# (C, H, W)

        # tb shape is (C, H, W)
        normalized_tb = np.empty_like(tb)

        for i, IR_name in enumerate(input_list):
            mean, std = channel_stats[IR_name]
            normalized_tb[i] = (tb[i] - mean) / (3*std)

        with torch.no_grad():
            output = inverse(unet_matthieu(torch.tensor(normalized_tb).unsqueeze(0)))  # Add batch dimension

        #save the output in data['cnn_output']
        data['cnn_output'] = output.detach().cpu().numpy().squeeze()

        save_path = os.path.join("/net/nfs/ssd3/cfrancoismartin/Projects/datasets/fused_dataset/dataset", year, file_name)

        np.save(save_path, data)
    end = time.time()
    print(f"Year {year} processed in {end - start:.2f} seconds", flush=True)
# new_data = np.load(save_path, allow_pickle=True).item()
# #plot data['rain_rate'], data['cnn_output']
# new_cnn_output = new_data['cnn_output']
# new_rain_rates = new_data['rain_rate']
# vmin, vmax = 0.1, 50
# norm = LogNorm(vmin=vmin, vmax=vmax)
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(new_rain_rates, cmap='jet', norm=norm)
# plt.title("Rain Rate")
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(new_cnn_output, cmap='jet',norm=norm)
# plt.title("CNN Output")
# plt.colorbar()

# plt.tight_layout()
# plt.savefig("./Projects/datasets/output.png")
# plt.show()

# print("CNN Output stats:")
# print("Mean:", new_cnn_output.mean())
# print("Min:", new_cnn_output.min())
# print("Max:", new_cnn_output.max())