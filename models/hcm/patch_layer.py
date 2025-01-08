import torch
import torch.nn as nn
import math

class TSMixerProjection(nn.Module):
    def __init__(self, seq_len, out_len, hidden_size, dropout):
        super().__init__()
        
        # Time mixing
        self.time_mixer = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Channel mixing
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Final projection
        self.output_proj = nn.Linear(seq_len, out_len)
        
    def forward(self, x):
        # x: [bs, vars, seq_len]
        
        # Time mixing
        res = x
        x = self.time_mixer(x)
        x = x + res
        
        # Channel mixing
        res = x
        x = x.transpose(1, 2)
        x = self.channel_mixer(x)
        x = x.transpose(1, 2)
        x = x + res
        
        # Project to output length
        x = self.output_proj(x)
        return x

class Cluster_wise_TSMixer(nn.Module):
    def __init__(self, n_cluster, n_vars, in_len, out_len, hidden_size, dropout, device):
        super().__init__()
        self.n_cluster = n_cluster
        self.n_vars = n_vars
        self.in_len = in_len
        self.out_len = out_len
        
        self.cluster_projectors = nn.ModuleList([
            TSMixerProjection(
                seq_len=in_len,
                out_len=out_len,
                hidden_size=hidden_size,
                dropout=dropout
            ) for _ in range(n_cluster)
        ])

    def forward(self, x, assignments):
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.n_vars, self.out_len).to(x.device)
        
        for i in range(self.n_cluster):
            cluster_mask = assignments[:, i]
            if cluster_mask.sum() == 0:
                continue
                
            cluster_vars = x[:, cluster_mask.bool(), :]
            projected = self.cluster_projectors[i](cluster_vars)
            output[:, cluster_mask.bool(), :] = projected
            
        return output