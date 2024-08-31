from diffusion.gaussian_diffusion import GaussianDiffusion

diffusion = create_diffusion(timestep_respacing="")  

diffusion.training_losses(model, x, t, model_kwargs)