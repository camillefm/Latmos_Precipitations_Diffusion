
from functools import partial
import numpy as np
import pandas as pd

import torch
from torch import optim, nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR

from src.Unet.Unet_backbone.attention import Attention, LinearAttention,Residual, PreNorm
from src.Unet.Unet_backbone.block import ResnetBlock
from src.Unet.Unet_backbone.time_embedding import SinusoidalPositionEmbeddings
from src.Unet.Unet_backbone.sample import Upsample, Downsample, default

from src.Sampling.diffusion_constants import DiffusionConstants
from src.Sampling.sample import q_sample

from src.Unet.loss import loss_function, TimestepsLoss

from src.Metrics.trainer_metrics import TrainerMetrics
from src.Metrics.eval_metrics import EvalMetrics

from src.Visualization.trainer_plot import TrainerPlot
from src.Visualization.eval_plot import EvalPlot

class Unet(pl.LightningModule):
    def __init__(self,config, logger= None):
        super().__init__()
        self.load_config(config)
        self.build_model()
                # Initialize diffusion constants for the given number of timesteps
        self.diffusion_constants = DiffusionConstants(self.timesteps, schedule_name=config['diffusion']['scheduler'], clip_max=config['diffusion']['clip_betas_max'])

        #innitialize trainer metrics
        self.trainer_metrics = TrainerMetrics(self, timesteps=self.timesteps, normalized=self.norm_rain, log1p=self.log_norm_rain)
        self.eval_metrics = EvalMetrics(self, timesteps=self.timesteps, normalized=self.norm_rain, log1p=self.log_norm_rain, number_of_images_plot=4)

        self.timesteps_loss = TimestepsLoss(10, timesteps=self.timesteps, loss_type=self.loss_type)
        assert logger is None or config['eval']['to_wandb'], "wandb requested but no logger detected"
        
            # Initialize trainer plotS
        self.trainer_plot = TrainerPlot(wandb_logger=logger, rain_valid_threshold=self.valid_data_threshold, tb_normalised_nb_sigmas=config['transforms']['normalize_tb_nb_of_sigmas'])
        self.eval_plot = EvalPlot(wandb_logger=logger, rain_valid_threshold=self.valid_data_threshold, tb_normalised_nb_sigmas=config['transforms']['normalize_tb_nb_of_sigmas'])

    
    def load_config(self, config):

        # experiment number
        self.num_experiment = config['num_experiment']

        # Load configuration parameters
        self.channels = config['model']['n_channels']   
        self.dim_mults = config['model']['dim_mults']
        self.dim = config['model']['init_dim']
        self.out_channels = config['model']['out_dim']
        self.resnet_block_groups = config['model']['resnet_block_groups']
        self.image_size = config['model']['image_size']

        # Training configuration
        self.lr = config['training']['learning_rate']
        self.batch_size = config['training']['batch_size']

        # Data configuration
        self.valid_data_threshold = config['rain_value_threshold']
        self.log_norm_rain = config['transforms']['rain_log_normalization']
        self.norm_rain = config['transforms']['normalize_rain']
        self.normalize_rain_nb_of_sigmas = config['transforms']['normalize_rain_nb_of_sigmas']

        # Loss configuration
        self.loss_type = config['loss']['type']

        # Diffusion configuration
        self.timesteps = config['diffusion']['timesteps']
        self.cond_nb_channels = len(config['list_channels'])  # Number of conditioning channels
        self.cond_multiplier = config['diffusion']['cond_multiplier']


        # Learning rate scheduler configuration
        self.scheduler_type = config['scheduler']['type']
        self.lr_gamma = config['scheduler']['gamma']
        self.lr_step_size = config['scheduler']['step_size']

        # Metrics configuration
        self.compute_metrics_rate = config['metrics']['compute_rate']
        self.nb_batch_metrics = config['metrics']['nb_elements_for_metrics']//self.batch_size

        #plotting eval
        self.to_wandb = config['eval']['to_wandb']
        self.eval_as_png = config['eval']['as_png']

        # Other attributes
        self.batch_to_plot = None
        self.resume_from_checkpoint = config['training']['resume_from_checkpoint']   


        #load mean and std for rain normalization
        if self.norm_rain:
            #read csv file with mean and std
            rain_stats = pd.read_csv(config['norm_csv_directory'] + f"e{self.num_experiment}_csv_rain.csv")
            self.mean_r = torch.tensor(rain_stats['mean'], dtype=torch.float32)
            self.std_r = torch.tensor(rain_stats['std'], dtype=torch.float32)

        if self.eval_as_png:
            self.png_directory = config['result_directory']
    
    def build_model(self):

        input_channels = self.channels + self.cond_nb_channels * self.cond_multiplier 

        self.init_conv = nn.Conv2d(input_channels, self.dim, 1, padding=0,padding_mode='reflect') # changed to 1 and 0 from 7,3

        # Define dimensions for each UNet block
        dims = [self.dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Partial function for ResNet blocks
        block_klass = partial(ResnetBlock, groups=self.resnet_block_groups)

        # Time embedding dimension
        time_dim = self.dim * 4

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dim),
            nn.Linear(self.dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Downsampling layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1,padding_mode='reflect'),
                    ]
                )
            )

        # Middle blocks (bottleneck)
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling layers
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1,padding_mode='reflect'),
                    ]
                )
            )

        # Set output channels
        self.out_channels = default(self.out_channels, self.channels)

        # Final blocks
        self.final_res_block = block_klass(self.dim * 2, self.dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(self.dim, self.out_channels, 1,padding_mode='reflect')

    def forward(self, x, y, time):

        # Repeat conditioning and concatenate to input
        for _ in range(self.cond_multiplier):
            x = torch.cat((x, y), dim = 1)  

        # Initial convolution
        x = self.init_conv(x)
        r = x.clone()
        
        t = self.time_mlp(time)
        
        h = []
        
        # Downsampling path
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        # Middle blocks
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # Upsampling path
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            
            x = upsample(x)
        
        # Final blocks
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x= self.final_conv(x)

        return x
    
    def configure_optimizers(self):
        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # Set up learning rate scheduler if specified
        if self.scheduler_type == "step":
            scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        else:
            return optimizer

    def _shared_step(self, batch, val = False):

        # Unpack the batch
        tb, r, rq = batch
        tb = tb.to(self.device)
        r = r.to(self.device)
        rq = rq.to(self.device)
        # Check for NaNs in the input tensors
        assert not torch.isnan(tb).any(), "/!\ NaNs in tb"
        assert not torch.isnan(r).any(), "/!\ NaNs in r"


        batch_size = tb.shape[0]

        # Randomly sample diffusion timesteps for each sample in the batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(r, dtype=torch.float32)
        noisy_r = q_sample(self, x_start=r, t=t, timesteps=self.timesteps, noise=noise)

        predicted_noise = self(noisy_r, tb, t)

        if val:
            self.timesteps_loss.update(noise, predicted_noise,t)
        loss = loss_function(noise, predicted_noise, loss_type=self.loss_type)
        assert not torch.isnan(loss).any(), "/!\ NaNs in loss"

        r = r * self.std_r.to(self.device) * self.normalize_rain_nb_of_sigmas + self.mean_r.to(self.device) if self.norm_rain else r
        return loss, tb, r, rq
    
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch)


            # Log training loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, tb, r, rq = self._shared_step(batch, val=True)
        
        

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute and update metrics at the specified rate (every N epochs)
        
        if (self.current_epoch % self.compute_metrics_rate == 0 and batch_idx < self.nb_batch_metrics) or self.resume_from_checkpoint:
            self.trainer_metrics.update_batch(tb, r,compute_metrics_batch=True)
            if batch_idx ==0:
                
                # Randomly select a batch index to plot
                self.random_batch_idx = np.random.randint(0, self.nb_batch_metrics)
                self.batch_to_plot = (tb, r, self.trainer_metrics.sampled_images, rq)
            
            if batch_idx == self.random_batch_idx:
                
                self.batch_to_plot = (tb, r, self.trainer_metrics.sampled_images, rq)

            # Store the batch to plot if this is the randomly selected batch index
        else:
            if batch_idx == self.random_batch_idx:
                self.trainer_metrics.update_batch(tb, r, compute_metrics_batch=False)
                self.batch_to_plot = (tb, r, self.trainer_metrics.sampled_images, rq)  
            
        return loss
    
    def on_validation_epoch_end(self):

        loss_timesteps_dict = self.timesteps_loss.compute()

        # Grouped loss metrics with consistent naming
        self.log_dict(
            {f"val/loss_interval/{k}": v for k, v in loss_timesteps_dict.items()},
            on_epoch=True,
            logger=True
        )

        self.timesteps_loss.reset()
        if self.current_epoch % self.compute_metrics_rate == 0 or self.resume_from_checkpoint:
            distance_metrics, anormality_metrics = self.trainer_metrics.compute_epoch_metrics()
            if self.to_wandb:
                self.trainer_plot.log_metrics(distance_metrics,self)
                self.trainer_plot.log_metrics(anormality_metrics,self)
            self.resume_from_checkpoint = False

        if self.to_wandb:
            tb_plot, r_plot, sampled_images_plot, rq_plot = self.batch_to_plot
            self.trainer_plot.plot_to_wandb(tb=tb_plot,r=r_plot,sampled_images=sampled_images_plot,rq=rq_plot,
                                        epoch=self.current_epoch,key=f"Validation Images, (random batch)")
            
            # Log the learning rate
            opt = self.optimizers()
            lr = opt.param_groups[0]["lr"]
            self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=False)

                # Reset metrics for the next epoch
        self.trainer_metrics.reset()
       


    #_____________________EVALUATION_____________________   


    def test_step(self, batch, batch_idx):
        _ , tb, r, rq = self._shared_step(batch)



        self.eval_metrics.update_batch(tb, r,rq)
           

    def on_test_epoch_end(self):
        print('on_test_epoch_end called')

        # Compute and log evaluation metrics
        distance_metrics, anormality_metrics, binary_metrics, pixel_metrics  = self.eval_metrics.compute_metrics()

        self.eval_plot.log_metrics(distance_metrics, name = "distance metrics" , to_wandb=self.to_wandb , as_png=self.png_directory + f"e{self.num_experiment}_distance_metrics.png")
        self.eval_plot.log_metrics(binary_metrics, name = 'binary metrics', to_wandb=self.to_wandb, as_png=self.png_directory + f"e{self.num_experiment}_binary_metrics.png")
        self.eval_plot.log_metrics(pixel_metrics, name = "pixel metrics", to_wandb=self.to_wandb, as_png=self.png_directory + f"e{self.num_experiment}_pixel_metrics.png")
        # self.eval_plot.log_metrics(anormality_metrics)
        # Dynamically get metric names from distance_metrics keys
        metrics = []
        for metric in distance_metrics.keys():
            if metric!= "early stopping rmse*(1-csi)":
                metrics.append((metric, f"best_images_{metric}"))
        for metric in distance_metrics.keys():
            if metric!= "early stopping rmse*(1-csi)":
                metrics.append((metric, f"worst_images_{metric}"))

        for metric, attr in metrics:
            list_image = getattr(self.eval_metrics, attr)
            key_prefix = "RainDiffusion_eval_best_" if "best" in attr else "RainDiffusion_eval_worst_"
            key = f"{key_prefix}{metric}"
            self.eval_plot.plot_images(list_image=list_image, epoch=self.current_epoch, key=key, to_wandb=self.to_wandb, as_png=self.png_directory + f"e{self.num_experiment}_{key}.png")

        self.eval_plot.plot_images(list_image=self.eval_metrics.list_random_images, epoch=self.current_epoch, key="RainDiffusion_eval_random_images", to_wandb=self.to_wandb, as_png=self.png_directory + f"e{self.num_experiment}_random_images.png")
        # Reset metrics for the next evaluation

        self.eval_plot.plot_histogram_distribution(rain_pixel_values=self.eval_metrics.rain_pixel_values,
                                                sampled_pixel_values=self.eval_metrics.sampled_pixel_values,
                                                key="Sampled vs Rain Pixel Values Histogram", to_wandb=self.to_wandb, as_png=self.png_directory + f"e{self.num_experiment}_histogram.png")
        
        self.eval_plot.plot_range_metrics(self.eval_metrics.range_metrics, key="RainDiffusion_eval_range_metrics", to_wandb=self.to_wandb, as_png=self.png_directory + f"e{self.num_experiment}_range_metrics.png")

        self.eval_plot.plot_confusion_matrix(self.eval_metrics.range_confusions, key="RainDiffusion_eval_confusion_matrix", to_wandb=self.to_wandb, as_png=self.png_directory + f"e{self.num_experiment}_confusion_matrix.png")

        self.eval_metrics.reset()

        
    
