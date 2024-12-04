import torch
from torch import nn 
import torch.nn.functional as F
import numpy as np

from .spatial_embedder import SpatialEmbedder, rbf_embed
from .expression_embedder import ExpressionEmbedder
from ..utils import normalize_layer, activation_fn
from ..layer import NeighborhoodRepresentation

class SpatialTEmbedder(nn.Module):

    def __init__(self,
                config):
        
        super(SpatialTEmbedder, self).__init__()
        self.config = config

        self.expression_embedder = ExpressionEmbedder(config)
        self.spatial_embedder = SpatialEmbedder(config)

        in_dim_conv = config['trimmed_exp_len']
        in_dim_mlp = config['trimmed_exp_len'] * config['expression_emb_dim']
        in_dim_att = config['expression_emb_dim']

        self.neighborhood_rep = NeighborhoodRepresentation(in_dim_conv, in_dim_mlp, in_dim_att, config)

    
    def forward(self, neighbor_exp, exp_cc, dist, cell_size, cell_shape, size, shape):
        neighborhood_emb = self.expression_embedder(neighbor_exp, True)
        cell_emb = self.expression_embedder(exp_cc, False)

        if self.config['tokenizer_type'] == 'gene_name':
            neighborhood_rep_emb, recon_loss = self.neighborhood_rep(neighborhood_emb)
        
            if self.config['cell_size']:
                cell_size = size.unsqueeze(-1)
                neighborhood_rep_emb = torch.cat([neighborhood_rep_emb, cell_size], dim=-1)
            
            if self.config['cell_shape']:
                neighborhood_rep_emb = torch.cat([neighborhood_rep_emb, cell_shape], dim=-1)
            
            spatial_embedding = self.spatial_embedder(dist, neighborhood_emb, 'gene_name')
            neighborhood_rep_emb = neighborhood_rep_emb + spatial_embedding

        elif self.tokenizer_type == 'sentencepiece':
            emb = self.spatial_embedder.spatial_sentencepiece(neighborhood_emb, dist)
        
        return neighborhood_rep_emb, recon_loss, cell_emb
    
    