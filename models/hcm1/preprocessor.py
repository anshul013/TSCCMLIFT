import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class HardClusterAssigner(nn.Module):
    """Module for hard clustering of channels"""
    def __init__(self, n_vars, num_clusters, method='kmeans', device='cpu'):
        super().__init__()
        self.n_vars = n_vars
        self.num_clusters = num_clusters
        self.method = method
        self.device = device
        
        # Initialize clusterer
        if method == 'kmeans':
            self.clusterer = KMeans(n_clusters=num_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Initialize cluster assignments
        self.register_buffer('cluster_assignments', torch.zeros(n_vars, dtype=torch.long))
        self.fitted = False
        self.inertia = None
        self.silhouette = None
        
    def extract_features(self, x):
        """Extract statistical features for clustering"""
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
            
        features = []
        for channel in range(x.shape[2]):
            channel_data = x[:, :, channel]
            features.append([
                np.mean(channel_data),
                np.std(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                np.max(channel_data),
                np.min(channel_data),
                np.mean(np.abs(np.diff(channel_data, axis=1)))
            ])
        return np.array(features)
    
    def fit(self, x):
        """Fit clustering on data"""
        features = self.extract_features(x)
        cluster_idx = self.clusterer.fit_predict(features)
        self.cluster_assignments = torch.tensor(cluster_idx, device=self.device)
        self.inertia = self.clusterer.inertia_
        if self.n_vars > self.num_clusters:
            self.silhouette = silhouette_score(features, cluster_idx)
        self.fitted = True
        return self.cluster_assignments
    
    def forward(self, x, if_update=False):
        """Forward pass returns cluster assignments"""
        if if_update or not self.fitted:
            self.fit(x)
        return self.cluster_assignments
    
    def get_metrics(self):
        """Get clustering metrics"""
        return {
            'inertia': self.inertia,
            'silhouette': self.silhouette
        } 