# from guided_diffusion.script_util import setup_model_diffusion

# model, diffusion = setup_model_diffusion(

from guided_diffusion.resample import create_named_schedule_sampler
diffusion = 0
sampler = create_named_schedule_sampler("uniform", diffusion, 1000)

indices, weights = sampler.sample(10, "cpu")
print(indices, weights)