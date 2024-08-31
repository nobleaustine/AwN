import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


from helper_functions.helpers import use_fused_attn

class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            p_or_not:bool = False
        ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 # scaling factor root(dk)
        self.fused_attn = use_fused_attn()
        self.p_or_not = p_or_not

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # if p_or_not:
        #     self.p = nn.Parameter(torch.ones(dim,dim))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, T, D = x.shape
        qkv = self.qkv(x).reshape(N, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(N, T, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):


    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
        ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor,y:torch.Tensor) -> torch.Tensor:
        N, T, D = x.shape
        kv = self.kv(x).reshape(N, T, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self.q(y).reshape(N, T, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = q[0]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(N, T, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SegAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
        ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # self.scale = self.head_dim ** -0.5 # scaling factor root(dk)
        self.i_scale = nn.Parameter((self.head_dim ** -0.5)*torch.ones(self.num_heads,1,self.head_dim))
        self.l_scale = nn.Parameter((self.head_dim ** -0.5)*torch.ones(self.num_heads,1,self.head_dim))
        # self.fused_attn = use_fused_attn()
        # self.p_or_not = p_or_not

        self.i_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.l_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.scalar_param = nn.Parameter(torch.tensor(0.9))
        
    # i: image, l: label, a: anti, n: noise, f: feature
    def forward(self, x: torch.Tensor,y: torch.Tensor) -> torch.Tensor:
        N, T, D = x.shape
        i_qkv = self.i_qkv(x).reshape(N, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        i_q, i_k, i_v = i_qkv.unbind(0)
        i_q, i_k = self.q_norm(i_q), self.k_norm(i_k)

        l_qkv = self.l_qkv(y).reshape(N, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        l_q, l_k, l_v = l_qkv.unbind(0)
        l_q, l_k = self.q_norm(l_q), self.k_norm(l_k)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p if self.training else 0.,
        #     )
        # else:

        i_scale = self.i_scale.softmax(dim=-1)
        i_q = i_q * i_scale
        n_attn = i_q @ l_k.transpose(-2, -1)

        n_attn = n_attn.softmax(dim=-1)
        an_attn = (n_attn.max(dim=-1, keepdim=True)[0] - n_attn).softmax(dim=-1)
        n_attn = self.attn_drop(n_attn)
        an_attn = self.attn_drop(an_attn)

        l_scale = self.l_scale.softmax(dim=-1)
        l_q = l_q * l_scale
        f_attn = l_q @ i_k.transpose(-2, -1)

        f_attn = f_attn.softmax(dim=-1)
        af_attn = (f_attn.max(dim=-1, keepdim=True)[0] - f_attn).softmax(dim=-1)
        f_attn = self.attn_drop(f_attn)
        af_attn = self.attn_drop(af_attn)

        x = (self.scalar_param*n_attn) @ l_v + ((1-self.scalar_param)*an_attn) @ i_v
        x = x.transpose(1, 2).reshape(N, T, D)

        y = (self.scalar_param*f_attn) @ i_v + ((1-self.scalar_param)*af_attn) @ l_v
        y = y.transpose(1, 2).reshape(N, T, D)

        return x,y


if __name__ == '__main__':
    x = torch.randn(2, 5, 32)
    y = torch.randn(2, 5, 32)
    att = SegAttention(32)
    a,b = att(x,y)
    

    