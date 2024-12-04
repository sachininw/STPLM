import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = device

        # Create a matrix of shape [max_len, d_model]
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = self.pe.unsqueeze(0)  # Shape [1, max_len, d_model]

        
    def forward(self, length):

        return self.pe[:, :length, :]


from .spatialT_embedder import SpatialTEmbedder