from Sampling.diffusion_constants import DiffusionConstants



import torch


# Utility function to extract values from a tensor 'a' at indices 't' and reshape to 'x_shape'
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device).float()


# Forward diffusion process: adds noise to the input data 'x_start' according to the diffusion schedule
def q_sample(denoise_model, x_start, t, noise=None, timesteps=300):
    diffusion_constants = denoise_model.diffusion_constants
    # Get mean and std for normalization from the denoise model


    x_start = x_start.float()
    # Get diffusion schedule constants
    sqrt_alphas_cumprod = diffusion_constants.sqrt_alphas_cumprod.float()
    sqrt_one_minus_alphas_cumprod = diffusion_constants.sqrt_one_minus_alphas_cumprod.float()
    if noise is None:
        # Generate random noise if not provided
        noise = torch.randn_like(x_start, dtype=torch.float32)
    else:
        noise = noise.float()
    
    # Extract the appropriate constants for the current timestep
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    # Return the noisy sample
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


@torch.no_grad()
def p_sample(model, x, tb, t, t_index, timesteps=300):
    #print_tensor_stats([(x, "x")], where="p_sample_start ")
    
    diffusion_constants = model.diffusion_constants
    betas = diffusion_constants.betas
    sqrt_one_minus_alphas_cumprod = diffusion_constants.sqrt_one_minus_alphas_cumprod
    sqrt_recip_alphas = diffusion_constants.sqrt_recip_alphas
    posterior_variance = diffusion_constants.posterior_variance
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean

    model_output = model(x, tb, t)

    #print_tensor_stats([(model_output, "model_output")], where="p_sample model_output")
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
    )
    
    #print_tensor_stats([(model_mean, "model_mean")], where="p_sample model_mean")

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample(
    model,
    tb,
    timesteps=300,
    normalized=True,
    log1p=False,
    init_noise=None,
    return_all_steps=False,
    nb_sigmas_rain=1
):
   

    device = next(model.parameters()).device
    mean_r, std_r = model.mean_r, model.std_r if normalized else (0, 1)
    batch_size = tb.shape[0]
    image_size = tb.shape[-1]
    channels = tb.shape[1]
    assert tb.shape[-2] == image_size, "tb is not a squar image"
    shape = (batch_size, 1, image_size, image_size)
    # start from pure noise or given init
    img = torch.randn(shape, device=device) if init_noise is None else init_noise.to(device)
    images = []

    for i in reversed(range(timesteps)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        img = p_sample(model=model, x=img, tb=tb, t=t, t_index=i, timesteps=timesteps)

        if return_all_steps:
            #apply unnormalization and unlog1p if needed
            denorm_img = img.cpu() * nb_sigmas_rain * std_r + mean_r if normalized else img
            denorm_img = torch.expm1(denorm_img) if log1p else denorm_img
            
            images.append(denorm_img.cpu())

    if return_all_steps:
        return images
    else:
        # apply unnormalization and unlog1p if needed
        
        final_img = img.cpu() * nb_sigmas_rain * std_r + mean_r if normalized else img
        final_img = torch.expm1(final_img) if log1p else final_img

        
        return final_img.cpu()