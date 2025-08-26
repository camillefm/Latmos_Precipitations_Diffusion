from Visualization.plot import plot_tensors
import wandb

class TrainerPlot():
    def __init__(self, wandb_logger= None, rain_valid_threshold=0.1, tb_normalised_nb_sigmas = 1):
        self.wandb_logger = wandb_logger
        self.rain_valid_threshold = rain_valid_threshold
        self.tb_normalised_nb_sigmas = tb_normalised_nb_sigmas

    def plot_to_wandb(self, tb=None, r=None, sampled_images=None, rq=None, epoch=None, key="RainDiffusion"):
                #replace zeros by nan in sampled_images
        sampled_images[sampled_images < self.rain_valid_threshold] = float('nan')

        assert all(x is not None for x in [tb, r, sampled_images, rq]), "All tensors must be provided for plotting."
        # Plotting the tensors
        
        numpy_plot = plot_tensors(tb_tensor=tb, rain_tensor=r, sampled_tensor=sampled_images, rq_tensor=rq, tb_normalised_nb_sigmas = self.tb_normalised_nb_sigmas)
        if self.wandb_logger is not None:
            self.wandb_logger.experiment.log({
                key: wandb.Image(numpy_plot),
                "epoch": epoch if epoch is not None else wandb.run.step,
            },
        )
        
    def log_metrics(self, metric_dict, model):
        for key, value in metric_dict.items():
            model.log(f"val_{key}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)




    