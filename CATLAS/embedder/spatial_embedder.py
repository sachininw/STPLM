import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from . import PositionalEncoding

def rbf_embed(distance, sigma=1):
          
        #Embedding distance using Radial Basis Function (RBF) kernels. 
        #RBF(d) = exp(-d^2 / 2 * sigma^2)

        return np.exp(-distance**2 / (2 * sigma**2))

class SpatialEmbedder_nn(nn.Module):
        def __init__(self, config):
            super(SpatialEmbedder_nn, self).__init__()
            self.config = config
            self.pe = PositionalEncoding(config['dim_neigh_rep'], config['trimmed_exp_len'])
            self.dist_embedder = nn.Embedding(config['neighborhood_radius'], config['dim_neigh_rep'])
        
        def forward(self, dist, x, tokenizer_type):
               
            if tokenizer_type=='gene_name':
                dist_embedding = self.dist_embedder(dist)

                return dist_embedding
                
            elif tokenizer_type=='sentencepiece':
                batch_size, num_tokens, _ = x.shape
                positional_encoding = self.pe.get_encoding(num_tokens)
                positional_encoding = positional_encoding.unsqueeze(1).repeat(batch_size, num_tokens, 1)

                return x+ positional_encoding

class SpatialEmbedder(nn.Module):
    def __init__(self, config):
        super(SpatialEmbedder, self).__init__()
        self.config = config
    
    def forward(self, dist, x, tokenizer_type):
         
        self.pe = torch.zeros(dist.size(0), dist.size(1), self.config['dim_neigh_rep'])
        position = dist.float()
        div_term = torch.exp(torch.arange(0, self.config['dim_neigh_rep'], 2).float() * (-math.log(10000.0) / self.config['dim_neigh_rep']))

        for i in range(dist.size(0)):
            self.pe[i, :, 0::2] = torch.sin(position[i, :].unsqueeze(1) * div_term)
            self.pe[i, :, 1::2] = torch.cos(position[i, :].unsqueeze(1) * div_term)
        
        return self.pe