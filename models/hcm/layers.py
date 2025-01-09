import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

class ClusterAssigner(nn.Module):
    """Module for hard clustering of variables"""
    def __init__(self, n_vars, num_clusters, method='kmeans', device='cpu'):
        super().__init__()
        self.n_vars = n_vars
        self.num_clusters = num_clusters
        self.method = method
        self.device = device
        
        # Initialize clusterer
        if method == 'kmeans':
            self.clusterer = KMeans(n_clusters=num_clusters)
        elif method == 'hierarchical':
            self.clusterer = AgglomerativeClustering(n_clusters=num_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Initialize cluster assignments and metrics
        self.register_buffer('cluster_assignments', torch.zeros(n_vars, dtype=torch.long))
        self.inertia = None
        self.silhouette = None
        
    def forward(self, x, if_update=False):
        """
        Args:
            x: Input tensor of shape [batch_size, n_vars, seq_len]
            if_update: Whether to update cluster assignments
        Returns:
            cluster_assignments: Tensor of shape [n_vars]
        """
        if if_update and self.training:
            # Compute variable representations and detach from computation graph
            var_features = self._compute_var_features(x).detach()
            
            # Update clusters using detached features
            cluster_idx = self.clusterer.fit_predict(var_features.cpu().numpy())
            self.cluster_assignments = torch.tensor(cluster_idx, device=self.device)
            
            # Update metrics
            if self.method == 'kmeans':
                self.inertia = self.clusterer.inertia_
                try:
                    self.silhouette = silhouette_score(var_features.cpu().numpy(), cluster_idx)
                except:
                    self.silhouette = None
        
        return self.cluster_assignments
    
    def _compute_var_features(self, x):
        """Compute features for clustering variables"""
        # Use multiple statistical features
        mean_features = x.mean(dim=[0, 2])  # [n_vars]
        std_features = x.std(dim=[0, 2])    # [n_vars]
        max_features = x.max(dim=2)[0].mean(dim=0)  # [n_vars]
        min_features = x.min(dim=2)[0].mean(dim=0)  # [n_vars]
        
        features = torch.stack([
            mean_features, 
            std_features,
            max_features,
            min_features
        ], dim=1)  # [n_vars, 4]
        
        return features
    
    def get_assignments(self):
        """Returns current cluster assignments"""
        return self.cluster_assignments.clone()
    
    def get_clustering_metrics(self):
        """Returns current clustering metrics"""
        metrics = {
            'inertia': self.inertia if self.method == 'kmeans' else None,
            'silhouette': self.silhouette
        }
        return metrics

# class RevIN(nn.Module):
#     """Reversible Instance Normalization in PyTorch."""

#     def __init__(self, num_features, axis=-2, eps=1e-5, affine=True):
#         super().__init__()
#         self.axis = axis
#         self.num_features = num_features
#         self.eps = eps
#         self.affine = affine

#         if self.affine:
#             self.affine_weight = nn.Parameter(torch.ones(num_features))
#             self.affine_bias = nn.Parameter(torch.zeros(num_features)) 

#     def forward(self, x, mode):
#         if mode == 'norm':
#             self._get_statistics(x)
#             x = self._normalize(x)
#         elif mode == 'denorm':
#             x = self._denormalize(x)
#         else:
#             raise NotImplementedError
#         return x

#     def _get_statistics(self, x):
#         self.mean = x.mean(dim=self.axis, keepdim=True).detach() # [Batch, Input Length, Channel]
#         self.stdev = torch.sqrt(x.var(dim=self.axis, keepdim=True, unbiased=False) + self.eps).detach()

#     def _normalize(self, x):
#         x = (x - self.mean) / self.stdev
#         if self.affine:
#             x = x * self.affine_weight
#             x = x + self.affine_bias
#         return x

#     def _denormalize(self, x):
#         if self.affine:
#             x = x - self.affine_bias
#             x = x / self.affine_weight
#         x = x * self.stdev
#         x = x + self.mean
#         return x
    # """Reversible Instance Normalization"""
    # def __init__(self, num_features: int, eps=1e-5, affine=True):
    #     super().__init__()
    #     self.num_features = num_features
    #     self.eps = eps
    #     self.affine = affine
        
    #     if self.affine:
    #         self.affine_weight = nn.Parameter(torch.ones(num_features))
    #         self.affine_bias = nn.Parameter(torch.zeros(num_features))
            
    # def forward(self, x, mode):
    #     if mode == 'norm':
    #         self._get_statistics(x)
    #         x = self._normalize(x)
    #     elif mode == 'denorm':
    #         x = self._denormalize(x)
    #     return x
        
    # def _get_statistics(self, x):
    #     dim2reduce = tuple(range(1, x.ndim-1))
    #     self.mean = x.mean(dim=dim2reduce, keepdim=True).detach()
    #     self.stdev = torch.sqrt(
    #         x.var(dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
    #     ).detach()
        
    # def _normalize(self, x):
    #     x = (x - self.mean) / self.stdev
    #     if self.affine:
    #         x = x * self.affine_weight[None, :, None]
    #         x = x + self.affine_bias[None, :, None]
    #     return x
        
    # def _denormalize(self, x):
    #     if self.affine:
    #         x = (x - self.affine_bias[None, :, None])
    #         x = x / self.affine_weight[None, :, None]
    #     x = x * self.stdev + self.mean
    #     return x 