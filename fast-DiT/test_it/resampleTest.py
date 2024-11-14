import sys
sys.path.append("/cluster/home/austinen/NTNU/AwN/")
from guided_diffusion.resample import create_named_schedule_sampler

diffusion = 0
sampler = create_named_schedule_sampler("uniform", diffusion, 1000)

indices, weights = sampler.sample(10, "cpu")

print(indices, weights)