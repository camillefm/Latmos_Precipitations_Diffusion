import torch.nn.functional as F

import numpy as np



def loss_function(noise, predicted_noise, loss_type='l2'):
    """
    Compute the loss between the true noise and the predicted noise.

    Args:
        noise (Tensor): Ground truth noise tensor.
        predicted_noise (Tensor): Predicted noise tensor.
        loss_type (str): Type of loss to use ('l1', 'l2', or 'huber').

    Returns:
        Tensor: Computed loss.
    """
    if loss_type == 'l1':
        return F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        return F.mse_loss(noise, predicted_noise,reduction='mean')
    elif loss_type == "huber":
        return F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError(f"Loss type '{loss_type}' is not implemented.")
    

def get_log_intervals(n, num_points=10):
    """
    Returns a sorted list of unique, integer log-distributed timesteps from 0 to n.
    
    Args:
        n (int): Maximum timestep.
        num_points (int): Number of log steps (more means finer spacing).

    Returns:
        List[int]: Log-distributed time steps.
    """
    # Avoid log(0) by starting from 1
    log_steps = np.logspace(start=0, stop=np.log10(n), num=num_points, dtype=int)
    unique_steps = sorted(set(log_steps))
    
    # Ensure 0 and n are included
    if 0 not in unique_steps:
        unique_steps = [0] + unique_steps
    if n not in unique_steps:
        unique_steps.append(n)
    
    return unique_steps

class TimestepsLoss:
    def __init__(self, nb_intervals, timesteps, loss_type):
        self.timesteps = timesteps
        self.loss_type = loss_type

        log_values = get_log_intervals(self.timesteps, nb_intervals)
        intervals = [(log_values[i], log_values[i+1]) for i in range(len(log_values) - 1)]
        self.losses_by_timesteps = {key: (0.0, 0) for key in intervals}  # (sum_loss, count)


    def update(self, noise, predicted_noise, t):
        for i in range(len(t)):
            noise_i = noise[i]
            predicted_noise_i = predicted_noise[i]
            t_i = t[i].item()  # make sure it's a scalar

            loss_i = loss_function(noise_i.unsqueeze(0), predicted_noise_i.unsqueeze(0), self.loss_type).item()

            for key in self.losses_by_timesteps:
                start, end = key
                if start <= t_i < end:
                    sum_loss, count = self.losses_by_timesteps[key]
                    self.losses_by_timesteps[key] = (sum_loss + loss_i, count + 1)
                    break  # each t_i belongs to only one interval

    def compute(self):
        loss_dict = {}
        for key, (total_loss, count) in self.losses_by_timesteps.items():
            key_str = f"{key[0]}-{key[1]}"
            mean_loss = total_loss / count if count > 0 else 0.0
            loss_dict[key_str] = mean_loss
        return loss_dict

    def reset(self):
        for key in self.losses_by_timesteps:
            self.losses_by_timesteps[key] = (0.0, 0)