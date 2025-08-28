import torch
import torch.nn.functional as F
from functools import partial

# Beta schedule functions
def cosine_beta_schedule(timesteps, s=0.008, clip_max=0.999):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, clip_max)

def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.04, timesteps)

def quadratic_beta_schedule(timesteps):
    return torch.linspace(0.0001**0.5, 0.04**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (0.04 - 0.0001) + 0.0001


class DiffusionConstants:
    """
    Precompute and expose constants used in a diffusion process.
    """

    def __init__(self, timesteps, schedule_name='linear', clip_max=0.999):
        self.timesteps = timesteps
        self.schedule_name = schedule_name

        # Use functools.partial to handle functions with additional arguments
        schedule_map = {
            'linear': linear_beta_schedule,
            'quadratic': quadratic_beta_schedule,
            'sigmoid': sigmoid_beta_schedule,
            'cosine': partial(cosine_beta_schedule, clip_max=clip_max)
        }

        if schedule_name not in schedule_map:
            raise ValueError(
                f"Invalid schedule name '{schedule_name}'. "
                f"Available options: {list(schedule_map.keys())}"
            )

        # Compute betas using selected schedule
        self.betas = schedule_map[schedule_name](timesteps)

        # Compute and expose constants
        self._compute_constants()

    def _compute_constants(self):
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )