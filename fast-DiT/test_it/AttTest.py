
"""
test case of SegAttention

x[2, 5, 32]
y[2, 5, 32]
dim = 32
num_heads = 8
head_dim = 4
i_scale[8,1,4] = 1/root(4) = 0.5
l_scale[8,1,4] = 1/root(4) = 0.5
i_qkv.weight[96,32], no bias
l_qkv.weight[96,32], no bias
q_norm = Identity()
k_norm = Identity()
scalar_param = 0.9

forward
i_qkv(x)[2, 5, 96]
i_qkv[3, 2, 8, 5, 4]
i_q[2, 8, 5, 4]
i_k[2, 8, 5, 4]
i_v[2, 8, 5, 4]
i_scale[8,1,4]

n_attn[2, 8, 5, 5]
n_attn.max(dim=-1, keepdim=True)[0][2, 8, 5, 1]
an_attn[2, 8, 5, 5]

self.scalar_param*n_attn[2, 8, 5, 5]
(self.scalar_param*n_attn)@l_v[2, 8, 5, 4]
((1-self.scalar_param)*an_attn) @ i_v[2, 8, 5, 4]
x[2, 5, 32]

"""

import sys
import torch

sys.path.append("/cluster/home/austinen/NTNU/AwN/SSD/others/fast-DiT/")
from layers.attention import SegAttention

x = torch.randn(2, 5, 32)
y = torch.randn(2, 5, 32)
att = SegAttention(32)
a,b = att(x,y)