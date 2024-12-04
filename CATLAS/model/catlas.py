import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import random

from ..embedder import SpatialTEmbedder
from ..encoder import Encoder
from ..layer import VanillaTransformerEncoderLayer, Modulator
from ..downstream import AnnotationDownStream

class CATLAS(nn.Module):
    def __init__(self,
                 config):
        super(CATLAS, self).__init__()
        self.config = config

        if config['activation'] == 'relu':
            self.act = nn.ReLU()
        elif config['activation'] == 'prelu':
            self.act = nn.PReLU()

        self.embedder = SpatialTEmbedder(config)
        self.neighborhood_encoder = Encoder(config['dim_neigh_rep'],
                               config['num_heads_cell_encoder'],
                               config['ff_hidden_dim_cell_encoder'],
                               config['encoder_out_dim'],
                               config['activation'],
                               config['dropout'],
                               config['num_layers_cell_encoder'],
                               last_fc = True,
                               norm=config['norm'],
                               model_type = config['transformer_type_cell_rep'],
                               norm_first = True,
                               causal = False #If input should be masked
                               )
        
        self.annotation_out = AnnotationDownStream(config)

        self.cell_demo_modular_hadamard = nn.Sequential(
            nn.Linear(4, config['encoder_out_dim'], dtype=torch.float64), #size = 1 , shape = 3 -> 1+3=4, #+4 for neighbor cell size and shape
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )

        self.cell_demo_modular = Modulator(config['cell_demo_in_dim'], config['cell_demo_hidden_dim'], config['encoder_out_dim'])
    
    def forward(self, neighbor_exp, neighbor_dist, neighbor_size, neighbor_shape, cell_exp, cell_size, cell_shape):

        neighborhood_rep_emb, recon_loss, cell_emb = self.embedder(neighbor_exp, cell_exp, neighbor_dist, cell_size, cell_shape, neighbor_size, neighbor_shape)

        neighborhood_rep = self.neighborhood_encoder(neighborhood_rep_emb)

        if self.config['cell_demo_modular']:

            shape_size_cc = torch.cat([neighbor_size.unsqueeze(-1).double(), neighbor_shape.double()], dim=-1)

            if self.config['cell_demo_modular_type'] == 'hadamard':
            #get hadamard product of center cell context and out
                cell_demo_modulation = self.cell_demo_modular_hadamard(shape_size_cc)
                neighborhood_rep = neighborhood_rep * (cell_demo_modulation.unsqueeze(1).repeat(1, neighborhood_rep.size(1), 1))

            elif self.config['cell_demo_modular_type'] == 'non_linear_feature_modulation':
                # size_cc or shape_cc
                neighborhood_rep = self.cell_demo_modular(neighborhood_rep, shape_size_cc)
            

        if self.config['pipeline'] == 'cell_annotation':

            output,recon_loss_dec, ls_reg  = self.annotation_out(cell_emb, neighborhood_rep)
            
            return output, recon_loss, recon_loss_dec, ls_reg

