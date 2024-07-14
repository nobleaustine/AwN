import torch
from diffusion import create_diffusion
from models.AWN import DiT_models


input = torch.randn(10,1,512,512)
y = torch.randn(10,1,512,512)
# y = torch.randn(10,1,512,512)
# t = torch.randn(10)
# ty = t* 100
# t = t.to(torch.int)
# t = torch.abs(t)


model = DiT_models["DiT-XL/2"](
input_size=512
)
diffusion = create_diffusion(timestep_respacing="") 
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

model.train()  # important! This enables embedding dropout for classifier-free guidance
t = torch.randint(0, diffusion.num_timesteps, (10,))
loss_dict = diffusion.training_losses(model,input, y,t)


import sys
sys.path.append("./")
print(sys.path)
# from DiT.diffusion import create_diffusion
# import torch as th

def model(x,t,y):
    print("x:",x.shape)
    print("t:",t.shape)
    print("y:",y.shape)
    mean = th.randn_like(x)
    var = th.randn_like(x)
    res = th.cat([mean,var],dim=1)
    return res

# diffusion = create_diffusion(timestep_respacing="")

# x = th.randn(10,1,512,512)
# y = th.randn(10,1,512,512)
# t = th.randint(0, diffusion.num_timesteps, (x.shape[0],))

# loss = diffusion.training_losses(model, x, t, dict(y=y))