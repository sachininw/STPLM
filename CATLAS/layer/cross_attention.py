import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 config,
                 eps=1e-6):
        super(CrossAttentionLayer, self).__init__()
        self.config = config
        self.eps = eps
        self.dropout = nn.Dropout(config['dropout'])


        if config['activation'] == 'relu':
            self.act = nn.ReLU()
        elif config['activation']  == 'prelu':
            self.act = nn.PReLU()

        if config['encoder_out_dim'] % config['n_heads_cross_att'] != 0:
            raise ValueError(f"dim {config['encoder_out_dim']} should be divisible by n_heads {config['encoder_out_dim']}")
        
        self.query_projection = nn.Linear(config['encoder_out_dim'], config['encoder_out_dim'], dtype=torch.float64)
        self.key_projection = nn.Linear(config['encoder_out_dim'], config['encoder_out_dim'], dtype=torch.float64)
        self.value_projection = nn.Linear(config['encoder_out_dim'], config['encoder_out_dim'], dtype=torch.float64)

        self.ff_block = nn.Sequential(nn.Linear(config['encoder_out_dim'], config['ff_hid_dim_cross_att'], dtype=torch.float64),
                                      nn.GELU(),
                                      nn.Dropout(config['dropout']),
                                      nn.Linear(config['ff_hid_dim_cross_att'], config['encoder_out_dim'], dtype=torch.float64),
                                      nn.Dropout(config['dropout']),
                                      )

        self.norm1 = nn.LayerNorm(config['encoder_out_dim'], dtype=torch.float64)
        self.norm2 = nn.LayerNorm(config['encoder_out_dim'], dtype=torch.float64)
    
    def save_attention_gradients(self, attn_grads):
        self.attn_grads = attn_grads
        
    def get_attention_gradients(self):
        return self.attn_grads
        
    def save_attention_map(self, attn_map):
        self.attn_map = attn_map
        
    def get_attention_map(self):
        return self.attn_map
        
    def transpose(self,x):
        new_x = x.size()[:-1] + (self.config['n_heads_cross_att'], int(self.config['encoder_out_dim'] / self.config['n_heads_cross_att']))
        x = x.view(*new_x)
        return x.permute(0, 2, 1, 3)
    
    def combine_heads(self, x, batch_size, seq_len, n_heads, dim):
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, n_heads * dim)
        
    def forward(self, x, context):
        print('x', x.size())
        print('context', context.size())

        batch_size, seq_len = x.size(0), x.size(1)
        x_query_layer = self.transpose(self.query_projection(x))
        context_key_layer = self.transpose(self.key_projection(context))
        context_value_layer = self.transpose(self.value_projection(context))

        #Take the scaled dot product between query and key
        attention_scores = torch.einsum('bhqd,bhkd->bhqk', [x_query_layer, context_key_layer]) / (self.config['encoder_out_dim']**0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)
        #Compute the attention output
        out1 = torch.einsum('bhqk,bhkd->bhqd', [attention_scores, context_value_layer]).reshape(batch_size, seq_len, self.config['encoder_out_dim'])
        #out1 = self.combine_heads(out1, batch_size, seq_len, self.n_heads, self.d_model)

        out1 = x + self.norm1(self.dropout(out1))
        out2 = self.norm2(self.ff_block(out1))
        out = out1 + self.dropout(out2)

        return out