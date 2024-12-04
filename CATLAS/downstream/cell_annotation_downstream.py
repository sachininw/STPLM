import torch
import torch.nn as nn
from ..layer import GMVAELatentLayer, Modulator, CrossAttentionLayer
from ..encoder import Encoder

class AnnotationDownStream(nn.Module):
    def __init__(self, 
                 config):
        super(AnnotationDownStream, self).__init__()
        self.config = config
        self.cross_att = CrossAttentionLayer(config)
        
        self.cell_exp_proj = nn.Linear(config['expression_emb_dim'], config['encoder_out_dim'], dtype=torch.float64)

        self.encoder_cell = Encoder(config['encoder_out_dim'],
                                    config['num_heads_cell_encoder'],
                                    config['encoder_out_dim'],
                                    config['encoder_out_dim'],
                                    config['activation'],
                                    config['dropout'],
                                    config['num_layers_cell_encoder'],
                                    last_fc = False,
                                    norm=config['norm'],
                                    model_type = 'flowformer',
                                    norm_first = True,
                                    causal = False  #If input should be masked
                                    )

        self.modulator_neighborhood = Modulator(config['encoder_out_dim'],
                                                config['neigh_modular_hidden_dim'],
                                                config['encoder_out_dim'])

        self.cross_att_layers = nn.ModuleList()
        cross_att_layer = self.cross_att
        for i in range(config['num_layers_cross_att']):
            self.cross_att_layers.append(
                cross_att_layer)
                    
        self.fc_out_pre_gmm= nn.Linear(config['trimmed_exp_len']*config['encoder_out_dim'], 
                                        config['fc_out_pre_gmm_dim'], 
                                        dtype=torch.float64)

        
        self.gmm_latent = GMVAELatentLayer(config['fc_out_pre_gmm_dim'], config['fc_out_dim'], config['cell_types'])

        
        self.fc_out = nn.Linear(config['fc_out_dim'], config['cell_types'], dtype=torch.float64)

        self.softmax = nn.Softmax(dim=-1) 

    def forward(self,
                cell_emb,
                neighborhood_rep):
        
        if self.config['annotation_downstream'] == 'transformer_encoder':
            neighborhood_rep = neighborhood_rep.view(-1, neighborhood_rep.size(1) * neighborhood_rep.size(2))
            neighborhood_rep = neighborhood_rep.double()
            

            cell_rep = self.encoder_cell(cell_emb)

            cell_rep = (self.modulator_neighborhood(cell_rep, neighborhood_rep, second=True)).double()
            
            output, recon_loss = self.gmm_latent(cell_rep)

            output =  self.softmax(self.fc_out(output))

            return output, recon_loss, 0
        
        elif self.config['annotation_downstream'] == 'contextual_reg':
            reg_weight = 0.5
            neighborhood_rep = (neighborhood_rep.view(-1, neighborhood_rep.size(1) * neighborhood_rep.size(2))).double()
            context = self.fc_out_pre_gmm(neighborhood_rep)

            cell_rep = self.encoder_cell(cell_emb)

            ls_reg = reg_weight * torch.mean((cell_rep - context) ** 2)

            cell_rep = cell_rep.double()
            
            output, recon_loss = self.gmm_latent(cell_rep)

            output =  self.softmax(self.fc_out(output))

            return output, recon_loss, ls_reg
        
        elif self.config['annotation_downstream'] == 'cross_attn':
            neighborhood_rep = neighborhood_rep.double()
            cell_emb = self.cell_exp_proj(cell_emb.double())
            cell_rep = self.encoder_cell(cell_emb)
            
            for i in range(self.config['num_layers_cross_att']):
                output = self.cross_att_layers[i](cell_rep, neighborhood_rep)

            '''
            Instead of the below, use a recurrent layer to combine the outputs of the cross-attention layers
            rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            _, (hidden_state, _) = rnn(attention_output) # hidden_state shape: (1, batch_size, hidden_dim)
            pooled_output = hidden_state.squeeze(0)  # Resulting shape: (batch_size, hidden_dim)

            because, flattening like below
            preserves all information but increases the input size for the classification layer, which may lead to overfitting.
            '''

            output = (output.view(-1, output.size(1) * output.size(2))).double()
            output = self.fc_out_pre_gmm(output)
            output, recon_loss = self.gmm_latent(output)

            output =  self.softmax(self.fc_out(output))

            return output,recon_loss,0