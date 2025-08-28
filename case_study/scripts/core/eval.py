import torch
import wandb
from pytorch_lightning.loggers import WandbLogger   

from core.src.Unet.unet import Unet
from core.src.Dataset.dataset import build_dataloaders

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Subset
import os


def eval_ddpm_model(config, debug_run: bool = False, checkpoint_name: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name ="eval e"+ str(config['num_experiment']) 
    print("cuda is available " ,torch.cuda.is_available())           # Should return True
    dataloader = build_dataloaders(config=config)
    test_loader = dataloader['test']

        # If debug_run is True, use a small subset of the data
    if debug_run:
        name += "_debug"
        print("Debug run enabled: using a small subset of the data.", flush=True)
        # Take only a few batches for debugging
        
        test_indices = list(range(min(16, len(test_loader.dataset))))

        test_loader = torch.utils.data.DataLoader(
            Subset(test_loader.dataset, test_indices),
            batch_size=test_loader.batch_size,
            shuffle=False,
            num_workers=0
        )
    name += str(config['eval_name'])

     # Check if training should resume from a previous checkpoint
    checkpoint_dir = config['checkpoint_directory']
    checkpoint_path = os.path.join(checkpoint_dir,checkpoint_name)
    result_dir = config['result_directory']
        #check if result folder exists, if not create it
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if config['eval']['to_wandb']:
        # Initialize Weights & Biases logging
        wandb.finish()  # Finish any previous runs
        wandb.login(key="380d7cb6473f438641f73f0650e92cdbf8b343f8")
        wandb.init(project="RainDiffusion", name=name, config=config)
        logger = WandbLogger(log_model=True)
    else: 
        logger = None
    # Create Trainer
    trainer = pl.Trainer(
        logger=logger,
        enable_progress_bar=False,
        precision=32,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        num_sanity_val_steps=0,
    )

    print("Loading model from checkpoint and evaluating...")
    # Evaluate the model using the test DataLoader
    model = Unet.load_from_checkpoint(checkpoint_path, config=config, logger= logger, device = device)


    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    trained_epoch = checkpoint['epoch']

    print(f"Model loaded from {checkpoint_path} at epoch {trained_epoch}", flush=True)

    del checkpoint  # Free memory

    trainer.test(model, dataloaders=test_loader)