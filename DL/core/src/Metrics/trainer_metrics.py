from src.Metrics.metrics_function import distance_metrics, anormality_metrics, update_dictionary
from src.Sampling.sample import sample
import torch

class TrainerMetrics:
    def __init__(self, model, timesteps, normalized=False, log1p=False):
        self.model = model
        self.timesteps = timesteps
        self.normalized = normalized
        self.log1p = log1p

        # Determine model device
        self.device = next(model.parameters()).device

        self.reset()

    def update_batch(self, tb, r, compute_metrics_batch = True):

        self.sampled_images = sample(self.model, tb, self.timesteps, self.normalized, self.log1p, init_noise=None, return_all_steps=False).to(self.device)
        #put negative values to 0
        self.sampled_images = torch.clamp(self.sampled_images, min=0)
        batch_size = tb.shape[0]
        if compute_metrics_batch:
            for i in range(batch_size):
                rain_i = r[i]
                sample_i = self.sampled_images[i]

                # Move tensors to CPU and detach before metric computation
                rain_i = rain_i.detach().cpu()
                sample_i = sample_i.detach().cpu()

                distance_metrics_result = distance_metrics(rain=rain_i, sampled_image=sample_i)
                anormality_metrics_result = anormality_metrics(sampled_image=sample_i)

                
                update_dictionary(self.distance_epoch, distance_metrics_result)
                
            
                update_dictionary(self.anormality_epoch, anormality_metrics_result)

                self.number_of_samples += 1

    def compute_epoch_metrics(self):
        for key in self.distance_epoch:
            self.distance_epoch[key] /= self.number_of_samples

        return self.distance_epoch, self.anormality_epoch


    def reset(self):
        self.distance_epoch = {}
        self.anormality_epoch = {}
        self.number_of_samples = 0
        self.sampled_images = None

   
    
