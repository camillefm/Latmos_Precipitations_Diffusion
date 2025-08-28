from core.src.Visualization.plot import plot_tensors, plot_histogram, plot_heatmap

import wandb
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

class EvalPlot():
    def __init__(self, wandb_logger= None, rain_valid_threshold=0.1, tb_normalised_nb_sigmas = 1):
        self.wandb_logger = wandb_logger
        self.rain_valid_threshold = rain_valid_threshold
        self.tb_normalised_nb_sigmas = tb_normalised_nb_sigmas

    def plot_images(self, list_image, epoch=None, key="RainDiffusion_eval", to_wandb=True, as_png=None):

        tb, r, sampled_images, rq = self.from_list_tuple_to_tensors(list_image)

        #replace zeros by nan in sampled_images
        sampled_images[sampled_images < self.rain_valid_threshold] = float('nan')


        assert all(x is not None for x in [tb, r, sampled_images, rq]), "All tensors must be provided for plotting."
        # Plotting the tensors
        numpy_plot = plot_tensors(tb_tensor=tb, rain_tensor=r, sampled_tensor=sampled_images, rq_tensor=rq, tb_normalised_nb_sigmas = self.tb_normalised_nb_sigmas)

        if to_wandb:
            self.wandb_logger.experiment.log({
                key: wandb.Image(numpy_plot),
                
            },)
            
        if as_png is not None:
            # Save the image to a PNG file
            img = Image.fromarray(numpy_plot)
            img.save(as_png)
            print(f"Image saved to {as_png}")
       
            
    def from_list_tuple_to_tensors(self, list_tuple):
        """
        Convert a list of tuples to tensors.
        Each tuple contains (tb, r, sampled_image, rq).
        Filters out None, empty, or scalar tensors before stacking.
        """
        tb_list, r_list, sampled_images_list, rq_list = zip(*list_tuple)

        def _filter_and_stack(tensor_list, name):
            # On ne garde que les tenseurs valides
            filtered = [
                t for t in tensor_list
                if isinstance(t, torch.Tensor) and t.numel() > 0 and t.dim() > 1
            ]
            if len(filtered) == 0:
                raise ValueError(f"No valid tensors found in {name}_list. Check data generation.")
            try:
                return torch.stack(filtered)
            except RuntimeError as e:
                shapes = [tuple(t.shape) for t in filtered]
                raise RuntimeError(f"Cannot stack tensors in {name}_list, got shapes {shapes}") from e

        tb_tensor = _filter_and_stack(tb_list, "tb")
        r_tensor = _filter_and_stack(r_list, "r")
        sampled_images_tensor = _filter_and_stack(sampled_images_list, "sampled_images")
        rq_tensor = _filter_and_stack(rq_list, "rq")

        return tb_tensor, r_tensor, sampled_images_tensor, rq_tensor


    
    def log_metrics(self, metrics_dict,name="Eval metrics", to_wandb=True, as_png=None):
        if to_wandb:
            table = wandb.Table(columns=["Metric", "Value"])
            for metric_name, metric_value in metrics_dict.items():
                table.add_data(metric_name, metric_value)

            wandb.log({name: table})
        if as_png is not None:
            # Create a DataFrame from the metrics dictionary
            df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
            # Plot DataFrame as a table using matplotlib
            fig, ax = plt.subplots(figsize=(1.5 * len(df.columns), 0.5 * len(df) + 1))
            ax.axis("off")
            mpl_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(10)
            mpl_table.scale(1.2, 1.2)

    def plot_histogram_distribution(self, rain_pixel_values, sampled_pixel_values, key="RainDiffusion_eval_histogram", to_wandb=True, as_png=None):
        image = plot_histogram(rain_pixel_values, sampled_pixel_values)
        if to_wandb:
            self.wandb_logger.experiment.log({
                key: wandb.Image(image),
            })
        
        if as_png is not None:
            # Save the image to a PNG file
            img = Image.fromarray(image)
            img.save(as_png)
            print(f"Histogram saved to {as_png}")


    def plot_range_metrics(self, range_metrics, key="RainDiffusion_eval_range_metrics", to_wandb=True, as_png=None):
        """
        Log range metrics (nested dict) to wandb as a table and optionally save as PNG.

        Parameters:
            range_metrics: dict[str, dict[str, float]]
            key: name for the table in wandb
            to_wandb: whether to log the table to Weights & Biases
            png_path: optional file path to save the table as a PNG image
        """
        # Get all unique metric names across all rain types
        all_metrics = sorted({metric for m in range_metrics.values() for metric in m.keys()})
        
        # Create DataFrame
        df = pd.DataFrame.from_dict(range_metrics, orient="index")
        df = df.reindex(columns=all_metrics)  # Ensure consistent column order
        df.index.name = "rain_type"
        df.reset_index(inplace=True)  # Make 'rain_type' a column

        df[df.columns[1:]] = df[df.columns[1:]].round(3)

        if to_wandb:
            # Define columns and create wandb Table
            columns = df.columns.tolist()
            table = wandb.Table(columns=columns)
            for _, row in df.iterrows():
                table.add_data(*row.values.tolist())
            wandb.log({key: table})

        if as_png is not None:
            # Plot DataFrame as a table using matplotlib
            fig, ax = plt.subplots(figsize=(1.5 * len(df.columns), 0.5 * len(df) + 1))
            ax.axis("off")
            mpl_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(10)
            mpl_table.scale(1.2, 1.2)  # Adjust cell scaling

            plt.tight_layout()
            plt.savefig(as_png, dpi=300)
            plt.close(fig)

            


    def plot_confusion_matrix(self, range_confusions, key="RainDiffusion_eval_confusion_matrix", to_wandb=True, as_png=None):
        """
        Log confusion matrix (nested dict) to wandb as a table.

        Parameters:
            range_confusions: dict[str, dict[str, float]]
            key: name for the table in wandb
        """

        image = plot_heatmap(range_confusions, title="Rain Diffusion Confusion Metrics Heatmap", cmap="Blues", fmt=".2f")

        if to_wandb:
            # Log the image to wandb
            self.wandb_logger.experiment.log({
                key: wandb.Image(image),
            })  

        if as_png is not None:
            # Save the image to a PNG file
            
            img = Image.fromarray(image)
            img.save(as_png)
            print(f"Confusion matrix saved to {as_png}")




        


    