import torch.nn as nn
import torch
import numpy as np
import random
import os
import logging

def set_seed(rndseed, cuda: bool = True, extreme_mode: bool = False):
    os.environ["PYTHONHASHSEED"] = str(rndseed)
    random.seed(rndseed)
    np.random.seed(rndseed)
    torch.manual_seed(rndseed)
    if cuda:
        torch.cuda.manual_seed(rndseed)
        torch.cuda.manual_seed_all(rndseed)
    if extreme_mode:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # dgl.seed(rndseed)
    # dgl.random.seed(rndseed)
    logging.info(f"Setting global random seed to {rndseed}")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
        
def normalize_layer(name, n, h=4):
    if name == 'layernorm':
        return nn.LayerNorm(n, dtype=torch.float64)
    elif name == 'batchnorm':
        return nn.BatchNorm1d(n, dtype=torch.float64)
    elif name == 'groupnorm':
        return nn.GroupNorm(h, n, dtype=torch.float64)
    elif name == 'rmsnorm':
        return nn.RMSNorm(n, dtype=torch.float64)
    else:
        raise NotImplementedError(f"{name} is not supported.")

def activation_fn(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'prelu':
        return nn.PReLU()
    elif name  == 'elu':
        return nn.ELU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not supported.")