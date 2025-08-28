import torch
import os
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.Unet.unet import Unet
from src.Dataset.dataset import build_dataloaders

import pytorch_lightning as pl

from torch.utils.data import Subset

def train_ddpm_model(config, debug_run: bool = False):

    # Set dataloaders using the provided configuration
    dataloader = build_dataloaders(config=config)
    train_loader = dataloader['train']
    val_loader = dataloader['val']


    # If debug_run is True, use a small subset of the data
    if debug_run:
        print("Debug run enabled: using a small subset of the data.", flush=True)
        # Take only a few batches for debugging
        train_indices = list(range(min(32, len(train_loader.dataset))))
        val_indices = list(range(min(12, len(val_loader.dataset))))
        train_loader = torch.utils.data.DataLoader(
            Subset(train_loader.dataset, train_indices),
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            Subset(val_loader.dataset, val_indices),
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=0
        )

    # Check if training should resume from a previous checkpoint
    resume_from_checkpoint = config['training']['resume_from_checkpoint']
    checkpoint_dir = f"/net/nfs/ssd3/cfrancoismartin/Projects/RainDiffusion/_Runs/checkpoints/cp_e{config['num_experiment']}/"
    checkpoint_name = config['training']['checkpoint_name']
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    #create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_csi = ModelCheckpoint(
        monitor="val_csi",
        mode="max",
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename="top-csi-{epoch:02d}-{csi:.4f}",
    )

    checkpoint_every_n =ModelCheckpoint(
    every_n_epochs=5,
    save_top_k=-1,
    dirpath=checkpoint_dir,
    filename="regular-{epoch:02d}"
    )   

    checkpoint_rmse = ModelCheckpoint(
        monitor="val_rmse",
        mode="min",
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename="top-rmse-{epoch:02d}-{rmse:.4f}",
    )

    checkpoint_val_loss = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        dirpath=checkpoint_dir,
        filename="top3-val_loss-{epoch:02d}-{val_loss:.4f}",
    )
    checkpoint_last = ModelCheckpoint(
    save_last=True,
    save_top_k=0,  # Don't save based on metric
    dirpath=checkpoint_dir,
    filename="last",  # Will create last.ckpt
    )

    checkpoint_early_stopping = ModelCheckpoint(
    
    save_top_k=3,  # Don't save based on metric
    dirpath=checkpoint_dir,
    filename="early_stop_best-{epoch:02d}",  
    monitor="val_early stopping rmse*(1-csi)",
)

    name = f"training e{config['num_experiment']}" + config['experiment_name']

    if debug_run:
        name = "debug " + name

    if config['eval']['to_wandb']:
        # Initialize Weights & Biases logging
        wandb.login(key="380d7cb6473f438641f73f0650e92cdbf8b343f8")
        wandb.init(project="RainDiffusion", name=name, config=config)
        

        logger = WandbLogger(log_model=True)
    else :
        logger = None

    # Set up the PyTorch Lightning trainer
    trainer = pl.Trainer(
        logger=logger,                                                   # Use WandbLogger for logging
        enable_progress_bar=False,                                       # Disable progress bar
        gradient_clip_val=1.0,                                               # Gradient clipping value
        precision=32,                                                         # Use mixed precision training
        max_epochs=12 if debug_run else config['training']['num_epochs'],       # Fewer epochs for debug
        accelerator="gpu" if torch.cuda.is_available() else "cpu",               # Use GPU acceleration
        callbacks=[checkpoint_every_n],  # Callbacks for checkpoints
    )

    # Initialize the model using the provided configuration
    model = Unet(config, logger)

    # Train or resume
    if resume_from_checkpoint and not debug_run:
        print("Resuming from checkpoint", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        trained_epoch = checkpoint['epoch']

        print(f"Model loaded from {checkpoint_path} at epoch {trained_epoch}", flush=True)

        del checkpoint  # Free memory

        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=checkpoint_path)
    else:
        print("Training from scratch", flush=True)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    if config['eval']['to_wandb']:
        wandb.finish()  # Finish the Wandb run