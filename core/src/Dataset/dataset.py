import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import pandas as pd

# You may need to import your custom transforms and normalization classes here:
from src.Dataset.transforms import SharedRandomCrop3D_XYRQ, LogNormalisation, MSNormalization, MinMaxNormalization, estimate_mean_std, save_mean_std_to_csv



class Npy_dataset(Dataset):
    """
    Custom Dataset for loading .npy binary files with support for input and target-specific transforms.

    Each file should have:
    - data: Infrared channels (inputs).
    - data['rain_rate']: Target value (regression with 99 classes).
    - data['rain_quality']: Quality of rain measurement.
    """
    def __init__(self,data_dir,input_list, file_list, shared_transform=None, input_only_transform=None,target_only_transform = None):
        """
        file_list: list
            List of file paths for the .npy files.
        shared_transform: callable, optional
            Transform to apply to both input and target data.
        input_only_transform: callable, optional
            Transform to apply only to input data.
        """
        self.data_dir = data_dir
        self.input_list = input_list
        self.file_list = file_list
        self.shared_transform = shared_transform
        self.input_only_transform = input_only_transform
        self.target_only_transform = target_only_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path, allow_pickle=True).item()

        try:
            tb = np.stack([data[IR_name] for IR_name in self.input_list]).astype(np.float32)# (C, H, W)
            r = np.array(data['rain_rate'], dtype=np.float32)
            rq = np.array(data['rain_quality'], dtype=np.float32)
        except KeyError as e:
            raise ValueError(f"Missing key in data: {e}")

        tb, r, rq = torch.tensor(tb), torch.tensor(r).unsqueeze(0), torch.tensor(rq).unsqueeze(0) #to get all on the same format

        # Apply shared transform if provided
        if self.shared_transform:
            tb, r, rq = self.shared_transform(tb, r, rq)

        if self.input_only_transform:
            tb = self.input_only_transform(tb)

        if self.target_only_transform:
            r = self.target_only_transform(r)

        return tb, r, rq

    def update_transforms(self, shared_transform=None, input_only_transform=None, target_only_transform=None):
        """Update the transforms after initialization."""
        if shared_transform is not None:
            self.shared_transform = shared_transform
        if input_only_transform is not None:
            self.input_only_transform = input_only_transform
        if target_only_transform is not None:
            self.target_only_transform = target_only_transform

    def to_csv(self, csv_file, split='Train'):
        """
        Write dataset metadata to a CSV file.
        
        csv_file: str
            Path to the CSV file to save.
        split: str
            Dataset split name (e.g., Train, Val, Test).
        """
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['split', 'file_number', 'file_path'])  # Header
            for idx, file_path in enumerate(self.file_list):
                writer.writerow([split, idx, file_path])

    def from_csv(self, csv_file):
        """
        Load dataset metadata from a CSV file.
        
        csv_file: str
            Path to the CSV file to load.
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} not found.")
        metadata = pd.read_csv(csv_file)
        self.file_list = metadata['path'].tolist() # Update file paths
        return self


def build_dataloaders(config):
    print("📦 Initializing datasets...")
    train_dataset = Npy_dataset(
        data_dir = config['data_directory'],
        input_list=config['list_channels'],
        file_list=[]

    ).from_csv(csv_file=config['split_directory']['train_split'])

    val_dataset = Npy_dataset(
        data_dir = config['data_directory'],
        input_list=config['list_channels'],
        file_list=[]

    ).from_csv(csv_file=config['split_directory']['val_split'])

    test_dataset = Npy_dataset(
        data_dir = config['data_directory'],
        input_list=config['list_channels'],
        file_list=[]

    ).from_csv(csv_file=config['split_directory']['test_split'])

    print("📦 Datasets initialized.")

    print("📦 Initializing dataloaders...")

    # Random crop
    if config['transforms']['random_crop'] == True:
        print("🗂️ Cropping ...")
        shared_crop = SharedRandomCrop3D_XYRQ(crop_size=config['model']['image_size'])

        train_dataset.update_transforms(shared_transform=shared_crop)
        val_dataset.update_transforms(shared_transform=shared_crop)
        test_dataset.update_transforms(shared_transform=shared_crop)

        # _______________RAIN  TRANSFORM____________________
    if config['transforms']['rain_log_normalization']:
        print("🗂️ Log(1+r) rain ...")
        log_normalizer = LogNormalisation()
        train_dataset.update_transforms(target_only_transform=log_normalizer)
        val_dataset.update_transforms(target_only_transform=log_normalizer)
        test_dataset.update_transforms(target_only_transform=log_normalizer)

    if config['transforms']['normalize_rain'] == "mean_std":
        print("🗂️ Normalizing rain to N(0,I) ...")
        # Estimate mean and std for normalization
        mean, std = estimate_mean_std(train_dataset, num_samples=10000, data_key='r', exclude_zeros=config['transforms']['normalize_rain_exclude_zeros'])
        #save mean and std to csv 
        num_experiment = config['num_experiment']
        save_mean_std_to_csv(config['norm_csv_directory'] + f"e{num_experiment}_csv_rain.csv", mean, std)

        target_normalizer = MSNormalization(mean=mean, std=std, nb_of_sigmas=config['transforms']['normalize_rain_nb_of_sigmas'])
        # Update train and val datasets
        train_dataset.update_transforms(target_only_transform=target_normalizer)
        val_dataset.update_transforms(target_only_transform=target_normalizer)
        test_dataset.update_transforms(target_only_transform=target_normalizer)
    elif config['transforms']['normalize_rain'] == "min_max":
        print("🗂️ Normalizing rain using min max ...")
        # Min-max normalization
        target_normalizer = MinMaxNormalization()
        train_dataset.update_transforms(target_only_transform=target_normalizer)
    
      # _______________TB  TRANSFORM____________________
    if config['transforms']['normalize_tb']== "mean_std":
        print("🗂️ Normalizing tb to N(0,I) ...")
        # Estimate mean and std for normalization
        mean, std = estimate_mean_std(train_dataset, num_samples=10000, data_key='tb', exclude_zeros=False)
        #save mean and std to csv 
        num_experiment = config['num_experiment']
        save_mean_std_to_csv(config['norm_csv_directory'] + f"e{num_experiment}_csv_tb.csv", mean, std)


        input_normalizer = MSNormalization(mean=mean, std=std, nb_of_sigmas=config['transforms']['normalize_tb_nb_of_sigmas'])
        # Update train and val datasets
        train_dataset.update_transforms(input_only_transform=input_normalizer)
        val_dataset.update_transforms(input_only_transform=input_normalizer)
        test_dataset.update_transforms(input_only_transform=input_normalizer)

    elif config['transforms']['normalize_tb'] == "min_max":
        print("🗂️ Normalizing tb using min max ...")
        # Min-max normalization
        input_normalizer = MinMaxNormalization()
        train_dataset.update_transforms(input_only_transform=input_normalizer)
        val_dataset.update_transforms(input_only_transform=input_normalizer)
        test_dataset.update_transforms(input_only_transform=input_normalizer)


    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Test samples: {len(test_dataset)}")

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'], 
            shuffle=config['transforms']['train']['shuffle'],
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']),
            
        "val": DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=config['transforms']['val']['shuffle'],
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']),

        "test": DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=config['transforms']['test']['shuffle'],
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory'])
    }
    return dataloaders

