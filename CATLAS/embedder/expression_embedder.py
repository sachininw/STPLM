import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from . import PositionalEncoding

class ExpressionEmbedder(nn.Module):
    def __init__(self, config):
        super(ExpressionEmbedder, self).__init__()
        self.config = config

        self.embedder = nn.Embedding(self.config['vocab_size'], self.config['expression_emb_dim'])
        self.pe = PositionalEncoding(self.config['expression_emb_dim'], self.config['trimmed_exp_len'], self.config['device'])
    
    def forward(self, x, neigh):
        if self.config['tokenizer_type'] == 'gene_name':
            exp_embedding = self.embedder(x)

            if neigh:
                batch_size, num_cells, _, _ = exp_embedding.size()
                positional_encoding = self.pe(self.config['trimmed_exp_len'])

                if not self.config['cat_exp_pos']:
                    positional_encoding = positional_encoding.repeat(batch_size, num_cells, 1, 1)
                    expression_embedding = exp_embedding + positional_encoding
                
                else:
                    positional_encoding = positional_encoding.repeat(batch_size, num_cells, 1)
                    expression_embedding = torch.cat((exp_embedding, positional_encoding), dim=-1)
            
            else:
                batch_size, exp_len, _ = exp_embedding.size()
                positional_encoding = self.pe(exp_len)

                if not self.config['cat_exp_pos']:
                    positional_encoding = positional_encoding.repeat(batch_size, 1, 1)
                    expression_embedding = exp_embedding + positional_encoding
                
                else:
                    positional_encoding = positional_encoding.repeat(batch_size, 1)
                    expression_embedding = torch.cat((exp_embedding, positional_encoding), dim=-1)

            return expression_embedding

        elif self.config['tokenizer_type'] == 'setencepiece':
            return self.embedder(x)

