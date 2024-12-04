import torch
import torch.nn as nn
import math
from ..embedder import PositionalEncoding
from . import FlowformerLayer

class Decoder(nn.Module):
    def __init__(self, 
                embed_dim, 
                num_heads, 
                num_layers_dec, 
                vocab_size, 
                max_exp_len_dec, 
                hid_dim, 
                activation,
                dropout=0.1, 
                transformer_type='vanilla', 
                device='cpu'):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_exp_len_dec+2, device) #+2 for <sos> and <eos> tokens in the target
        self.embed_dim = embed_dim
        self.max_exp_len_dec = max_exp_len_dec
        self.transformer_type = transformer_type
        self.device = device
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'prelu':
            self.act = nn.PReLU()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=vocab_size,
            dropout=dropout,
            activation=self.act,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers_dec
        )
        
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask ==1, float(0.0))
        return mask

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt = (self.embedding(tgt) * math.sqrt(self.embed_dim))

        memory = memory
        pos = self.pos_encoder(tgt.size(1))
        batch_size, seq_len, dim = tgt.size()
        pos = pos.repeat(batch_size, 1, 1)
        tgt = tgt + pos

        tgt = tgt.float()
        memory = memory.float()

        #if tgt_mask is None:
        #    tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(self.device)

        #tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
 
        if self.transformer_type=='vanilla':
            output = self.transformer_decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )


        #seq_len, batch_size, dim = output.size()
        #output = output.view(batch_size,  seq_len, dim)

        
        return output


