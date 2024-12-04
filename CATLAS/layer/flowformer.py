import torch
import torch.nn as nn
from . import AbstractTransformerLayer
from ..utils import normalize_layer

class Flow_Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.01, d_input=None, d_output=None, eps=1e-6):
        super(Flow_Attention, self).__init__()
        self.n_heads = n_heads
        if d_input is None:
            d_input = d_model
        if d_output is None:
            d_output = d_model
        
        self.query_projection = nn.Linear(d_input, d_model, dtype=torch.float64)
        self.key_projection = nn.Linear(d_input, d_model, dtype=torch.float64)
        self.value_projection = nn.Linear(d_input, d_model, dtype=torch.float64)
        self.out_projection = nn.Linear(d_model, d_output, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps
    
    def kernel_method(self, x):
        return torch.sigmoid(x)
    
    def dot_product(self, q, k, v):
        kv = torch.einsum('nhld,nhlm->nhdm', k, v)
        qkv = torch.einsum('nhld,nhdm->nhlm', q, kv)
        return qkv
    
    def forward(self, x):
        queries = keys = values = x
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1,2)
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)

        sink_incoming = 1.0 / (torch.einsum('nhld,nhd->nhl', queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum('nhld,nhd->nhl', keys + self.eps, queries.sum(dim=2) + self.eps))

        conserved_sink = torch.einsum('nhld,nhd->nhl', queries + self.eps,
                                        (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum('nhld,nhd->nhl', keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])

        x = (self.dot_product(queries * sink_incoming[:, :, :, None],
                              keys,
                              values * source_competition[:, :, :, None])
                              * sink_allocation[:, :, :, None]).transpose(1, 2)
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)

        return x

class Flow_Attention_Causal(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.05, d_input=None, d_output=None, eps=1e-6):
        super(Flow_Attention_Causal, self).__init__()
        self.n_heads = n_heads
        if d_input is None:
            d_input = d_model
        if d_output is None:
            d_output = d_model
        self.query_projection = nn.Linear(d_input, d_model, dtype=torch.float64)
        self.key_projection = nn.Linear(d_input, d_model, dtype=torch.float64)
        self.value_projection = nn.Linear(d_input, d_model, dtype=torch.float64)
        self.out_projection = nn.Linear(d_model, d_output, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps
    
    def kernel_method(self, x):
        return torch.sigmoid(x)
    
    def causal_dot_product(self, q, k, v):
        kv = torch.einsum('nhld,nhlm->nhldm', k, v)
        kv = torch.cumsum(kv, dim=2)
        qkv = torch.einsum('nhld,nhldm->nhlm', q, kv)
        return qkv
    
    def forward(self, x):
        queries = keys = values = x
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)

        sink_incoming = 1.0 / (torch.einsum('nhld,nhld->nhl', queries + self.eps, keys.cumsum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum('nhld,nhld->nhl', keys + self.eps, queries.cumsum(dim=2) + self.eps))
        normal = (((torch.arange(queries.shape[2])).float() + 1.0)).to(queries.device)[None, None, :]
        sink_incoming = sink_incoming * normal
        source_outgoing = source_outgoing * normal
        conserved_sink = torch.einsum('nhld,nhld->nhl', queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).cumsum(dim=2) + self.eps) / normal
        conserved_source = torch.einsum('nhld,nhld->nhl', keys + self.eps,
                                      (queries * sink_incoming[:, :, :, None]).cumsum(dim=2) + self.eps) / normal
        sink_allocation = torch.sigmoid(conserved_sink)
        conserved_source = torch.exp(conserved_source)
        source_competition = (conserved_source / conserved_source.cumsum(dim=-1)) * normal
        x = (self.causal_dot_product(queries * (sink_incoming[:, :, :, None] / normal[:, :, :, None]),
                                     keys,
                                     values * source_competition[:, :, :, None])
                                     * sink_allocation[:, :, :, None]).transpose(1, 2)
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x

class FlowformerLayer(nn.Module, AbstractTransformerLayer):
    def __init__(
        self,
        embed_dim,
        ff_hidden_dim,
        num_heads,
        dropout,
        norm = 'layernorm',
        norm_first = True,
        causal = False,
    ):

        super(FlowformerLayer, self).__init__()
        if not causal:
            self.self_attn = Flow_Attention(embed_dim, num_heads)
        else:
            self.self_attn = Flow_Attention_Causal(embed_dim, num_heads)
        self._ff_block = nn.Sequential(nn.Linear(embed_dim, ff_hidden_dim, dtype=torch.float64),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(ff_hidden_dim, embed_dim, dtype=torch.float64),
                                      nn.Dropout(dropout),
                                      )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = normalize_layer(norm, embed_dim)
        self.norm2 = normalize_layer(norm, embed_dim)
        self.norm_first = norm_first
        self.support_output_attentions = False
    
    def _sa_block(self, x):

        x = self.self_attn(x)
        return self.dropout1(x)[0, :, :]
    
    def forward(self, x, attn_mask=None, output_attentions=False):
        assert output_attentions == False, 'Output attentions not implemented'
        x = x.double()
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x
