import torch
import torch.nn as nn
from ..encoder import Encoder
from .gmmvi_latent import GMVAELatentLayer

class NeighborhoodRepresentation(nn.Module):
    def __init__(self,
                 in_dim_conv, 
                 in_dim_mlp,
                 in_dim_att,
                 config):
        super(NeighborhoodRepresentation, self).__init__()
        self.in_dim_conv = in_dim_conv
        self.in_dim_mlp = in_dim_mlp
        self.in_dim_att = in_dim_att
        self.config = config

        self.conv_layers = []
        self.dense_layers = []

        if config['activation'] == 'relu':
            act = nn.ReLU()
        elif config['activation'] == 'prelu':
            act = nn.PReLU()

        
        for i in range(config['num_layers_neigh_rep']):
            #out_dim = out_dim if i < num_layers -1 else 1
            self.conv_layers.append(nn.Conv2d(in_dim_conv, config['dim_neigh_rep'], kernel_size = (1,1)))
            self.conv_layers.append(act)
            in_dim_conv = config['dim_neigh_rep']
        
        self.conv_model = nn.Sequential(*self.conv_layers)

        for i in range(config['num_layers_neigh_rep']):
            #out_dim = out_dim if i < num_layers -1 else 1
            self.dense_layers.append(nn.Linear(in_dim_mlp, config['dim_neigh_rep']))
            self.dense_layers.append(act)
            in_dim_mlp = config['dim_neigh_rep']

        self.dense_model = nn.Sequential(*self.dense_layers)

        self.dense_model_att = nn.Sequential(
            nn.Linear(in_dim_att*config['trimmed_exp_len'], config['dim_neigh_rep'], dtype=torch.float64),
            act
        )

        self.gmm_latent = GMVAELatentLayer(in_dim_att*config['trimmed_exp_len'], config['dim_neigh_rep'], config['num_gmm_components_neigh_rep'])

        self.encoder = Encoder(in_dim_att,
                               config['num_heads_neigh_rep'],
                               config['dim_neigh_rep'],
                               config['dim_neigh_rep'],
                               config['activation'],
                               config['dropout'],
                               config['num_layers_neigh_rep'],
                               last_fc = False,
                               norm=config['norm'],
                               model_type=config['transformer_type_neigh_rep'],
                               norm_first=config['norm_first'],
                               causal = False
                          )
    
    def forward(self, x):
        batch_size, num_cells, seq, dim = x.shape
        x = x.reshape(batch_size*num_cells, seq, dim)

        if self.config['model_neigh_rep'] == 'conv':
            out = self.conv_model(x)
            out = out.squeeze(2)
        elif self.config['model_neigh_rep'] == 'mlp':
            x = x.view(batch_size * num_cells, -1) 
            out = self.dense_model(x)
            out = out.view(batch_size, num_cells, self.config['dim_neigh_rep'])
        
        elif self.config['model_neigh_rep'] == 'att':
            out = self.encoder(x)
            recon_loss = 0

            if self.config['dense_att_neigh_rep']:
                out = out.reshape(batch_size*num_cells, -1)
                out = self.dense_model_att(out)
                out = out.reshape(batch_size, num_cells, self.config['dim_neigh_rep']) 
            
            elif self.config['gmm_latent_neigh_rep']:
                out = out.reshape(batch_size*num_cells, -1) 
                out, recon_loss = self.gmm_latent(out)
                out = out.reshape(batch_size, num_cells, self.config['dim_neigh_rep']) 
            
            elif self.config['skip_dense_neigh_rep']:
                out = out.reshape(batch_size, num_cells, self.config['dim_neigh_rep'])

        return out, recon_loss