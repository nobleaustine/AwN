import collections.abc
import os
import torch
from torch import nn as nn
from enum import Enum
from itertools import repeat
import cProfile
import pstats
import re

# profile of the function
def show_profile(func,kwargs):
    profiler = cProfile.Profile()
    profiler.runcall(func,**kwargs)
    stats = pstats.Stats(profiler)
    stats.strip_dirs()  # Remove the file path
    stats.sort_stats('cumulative')
    stats.print_stats(lambda x: re.search(r'(<built-in|lib/python|dist-packages|site-packages', x[1]) is None)

# assert to print message
try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

# image dimention format
class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

# convert to NxCxHxW
def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x

# return function: create a tuple of length n
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

# create a tuple of length 2
to_2tuple = _ntuple(2)

def use_fused_attn(experimental: bool = False) -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0

    _HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    if 'TIMM_FUSED_ATTN' in os.environ:
        _USE_FUSED_ATTN = int(os.environ['TIMM_FUSED_ATTN'])
    else:
        _USE_FUSED_ATTN = 1  # 0 == off, 1 == on (for tested use), 2 == on (for experimental use)

    _EXPORTABLE = False
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0