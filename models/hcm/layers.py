import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HardClusterAssigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device):
        super(HardClusterAssigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        
        # Feature extractor
        self.linear = nn.Linear(seq_len, d_model)
        
        # Cluster centroids
        self.cluster_centroids = nn.Parameter(torch.randn(n_cluster, d_model))
        nn.init.kaiming_uniform_(self.cluster_centroids, a=math.sqrt(5))
        
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        
    def forward(self, x):
        # x: [bs, seq_len, n_vars]
        x = x.permute(0, 2, 1)  # [bs, n_vars, seq_len]
        
        # Extract features
        x_emb = self.linear(x)  # [bs, n_vars, d_model]
        x_emb = x_emb.mean(0)  # [n_vars, d_model]
        x_emb = self.l2norm(x_emb)
        
        # Calculate distances to centroids
        centroids_norm = self.l2norm(self.cluster_centroids)
        distances = -torch.mm(x_emb, centroids_norm.t())  # [n_vars, n_cluster]
        
        # Hard assignment using argmin
        assignments = torch.zeros_like(distances).scatter_(
            1, 
            distances.argmin(dim=1, keepdim=True),
            1.0
        )
        
        return assignments