import torch
import torch.nn as nn
import torch.nn.functional as F

class Modulator(nn.Module):

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super(Modulator, self).__init__()

        
        self.gamma_f1 = nn.Linear(in_dim, hidden_dim, dtype=torch.float64)
        self.gamma_f2 = nn.Linear(hidden_dim, out_dim, dtype=torch.float64)
        self.beta_fc1 = nn.Linear(in_dim, hidden_dim, dtype=torch.float64)
        self.beta_fc2 = nn.Linear(hidden_dim, out_dim, dtype=torch.float64)

    def forward(self, enc_out, feature_vec, second=False):
        if second:
            gamma = F.relu(self.gamma_f1(feature_vec))
            gamma = self.gamma_f2(gamma)

            beta = F.relu(self.beta_fc1(feature_vec))
            beta  = self.beta_fc2(beta)
            conditioned_output = (gamma * enc_out) + beta
            
            return conditioned_output

        else:

            gamma = F.relu(self.gamma_f1(feature_vec))
            gamma = self.gamma_f2(gamma)

            beta = F.relu(self.beta_fc1(feature_vec))
            beta  = self.beta_fc2(beta)

            gamma = gamma.unsqueeze(1).repeat(1, enc_out.size(1), 1)
            beta = beta.unsqueeze(1).repeat(1, enc_out.size(1), 1)
            conditioned_output = (gamma * enc_out) + beta
            
            return conditioned_output