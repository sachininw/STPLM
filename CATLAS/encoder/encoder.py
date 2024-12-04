import torch
import torch.nn as nn
from ..utils import activation_fn
from ..layer import VanillaTransformerEncoderLayer, FlowformerLayer

class Encoder(nn.Module):
    def __init__(self, 
                embed_dim,
                num_heads_encoder,
                ff_hidden_dim,
                embed_dim_enc_out,
                activation,
                dropout_rate,
                num_layers,
                last_fc = True,
                norm='layernorm',
                model_type = 'vanilla',
                norm_first = True,
                causal = False  #If input should be masked
                ):

        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.last_fc = last_fc
        
        self.fc_out = nn.Linear(embed_dim, embed_dim_enc_out, dtype = torch.float64)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'prelu':
            self.act = nn.PReLU()

        self.layers = nn.ModuleList()

        if model_type == 'cosformer':
            pass
        elif model_type == 'performer': 
            pass
        elif model_type == 'vanilla':
            TransformerLayer = VanillaTransformerEncoderLayer(embed_dim,
                                                              ff_hidden_dim,
                                                              num_heads_encoder,
                                                              dropout_rate,
                                                              activation
                                                             )

        elif model_type == 'flowformer':
            TransformerLayer = FlowformerLayer(embed_dim,
                                               ff_hidden_dim,
                                               num_heads_encoder,
                                               dropout_rate,
                                               norm,
                                               norm_first,
                                               causal)

        else:
            raise ValueError('Model type not recognized')
        
        for i in range(num_layers):
            self.layers.append(
                TransformerLayer)
    
    def forward(self, x, output_attentions=False):

        for i in range(self.num_layers):
            x = self.layers[i](x)

        x = x.double()
        
        if self.last_fc == True:
            x = self.act(self.fc_out(x))

        return x    