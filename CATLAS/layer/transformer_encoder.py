from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import torch
import copy
from typing import Optional, Any, Union, Callable
from ..utils import RMSNorm
from abc import ABC, abstractmethod
import math


class VanillaTransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, ff_hidden_dim, num_heads, dropout_rate, activation):
        super(VanillaTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True, dtype=torch.float64)
        self.linear1 = nn.Linear(embed_dim, ff_hidden_dim, dtype = torch.float64)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(ff_hidden_dim, embed_dim, dtype = torch.float64)
        
        self.norm1 = nn.LayerNorm(embed_dim, dtype=torch.float64)
        self.norm2 = nn.LayerNorm(embed_dim, dtype=torch.float64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'prelu':
            self.act = nn.PReLU()


    def forward(self, src):
        
        src = src.double()
        src1, attn_output_weights = self.self_attn(src, src, src)
        src1 = self.norm1(src1)
        src1 = src + self.dropout1(src1)

        src2 = self.dropout(self.act(self.linear1(src1)))
        src3 = self.dropout(self.linear2(src2))
        src3 = self.norm2(src3)
        src3 = src1 + self.dropout2(src3)
        
        return src3