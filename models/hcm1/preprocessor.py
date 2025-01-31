import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class HardClusterAssigner(nn.Module):
    """Module for hard clustering of channels"""
    def __init__(self, n_vars, num_clusters, method='kmeans', device='cuda', random_state=42):
        super().__init__()
        self.n_vars = n_vars
        self.num_clusters = num_clusters
        self.method = method
        self.device = device
        
        # Initialize clusterer with fixed random state
        if method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=num_clusters,
                random_state=random_state  # Add fixed random state
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Initialize cluster assignments on GPU
        self.register_buffer('cluster_assignments', 
                           torch.zeros(n_vars, dtype=torch.long, device=device))
        self.fitted = False
        self.inertia = None
        self.silhouette = None
        
    def extract_features(self, x):
        """Extract features for clustering"""
        # Keep computations on GPU where possible
        features = []
        x_numpy = x.cpu().numpy()  # Only convert to numpy for sklearn
        
        for i in range(self.n_vars):
            channel_data = x_numpy[:, :, i]
            features.append([
                np.mean(channel_data),
                np.std(channel_data),
                np.mean(np.abs(np.diff(channel_data, axis=1)))
            ])
        return np.array(features)
    
    def fit(self, x):
        """Fit clustering on data"""
        features = self.extract_features(x)
        cluster_idx = self.clusterer.fit_predict(features)
        
        # Move cluster assignments to GPU
        self.cluster_assignments = torch.tensor(
            cluster_idx, device=self.device, dtype=torch.long)
        
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