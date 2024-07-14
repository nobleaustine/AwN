
import torch

class LinearNoiseScheduler:

    def __init__(self,num_timesteps,beta_start,beta_end):
        self.num_timesteps = num_timesteps  # T: 1000
        self.beta_start = beta_start        # beta: 0.0001
        self.beta_end = beta_end            # beta: 0.002
        
        # betas: 0.001 - 0.002
        self.betas = torch.linspace(beta_start,beta_end,num_timesteps) # [b1,b2,....,bT]
        self.alphas = 1.0 - self.betas # [a1,a2,....aT]
        self.alpha_cum_prod = torch.cumprod(self.alphas,dim=0) # [a1.a2,a1.a2,a1.a2.a3,...,a1.a2.a3...aT]
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod) # [root(a1),root(a2),....root(aT)]
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1-self.alpha_cum_prod) # [root(1-a1),root(1-a2),....root(1-aT)]
    
    # forward process
    def add_noise(self,original,noise,t):
    
        original_shape = original.shape # batch*channel*height*width
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)# t.shape = b scaling factor at time t reshaping for b*1*1*1
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size) # standard deviation


        for _ in range(len(original_shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

            return original*sqrt_alpha_cum_prod + sqrt_one_minus_alpha_cum_prod*noise
        
    # backward process
    def sample_prev_timestep(self,xt,noise_pred,t):
        x0 = (xt -(self.sqrt_one_minus_alpha_cum_prod[t]*noise_pred))/self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0,-1.,1.) 

        mean = xt-((self.betas[t]*noise_pred)/(self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean/ torch.sqrt(self.alphas[t])

        if t == 0:
            return mean,x0
        else:
            variance = (1- self.alpha_cum_prod[t-1])/(1-self.alpha_cum_prod[t])
            variance = variance*self.betas[t]
            sigma = variance**0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma*z, x0
        
    






