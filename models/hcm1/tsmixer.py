import torch
import torch.nn as nn
from models.hcm1.preprocessor import HardClusterAssigner
from models.Rev_in import RevIN
from models.hcm1.blocks import TSMixerBlock

# class ClusterTSMixer(nn.Module):
#     """TSMixer model adapted for cluster-specific processing"""
#     def __init__(self, num_features, args):
#         super().__init__()
#         self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
#         self.model = TSMixerBlock(
#             in_len=args.seq_len,
#             out_len=args.pred_len,
#             d_ff=args.d_ff,
#             n_layers=args.n_layers,
#             enc_in=num_features,
#             dropout=args.dropout
#         ).to(self.device)
        
#     def forward(self, x):
#         return self.model(x)

class TSMixerH(nn.Module):
    """Hard Clustering variant of TSMixer"""
    def __init__(self, args):
        super().__init__()
        # Basic parameters
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.enc_in
        self.in_len = args.seq_len
        self.out_len = args.pred_len
        self.num_clusters = args.num_clusters
        self.d_ff = args.d_ff
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.enc_in = args.enc_in
        self.args = args
        
        # Normalization layer for full input
        self.rev_in = RevIN(num_features=args.enc_in)
        
        # Clustering module
        self.cluster_assigner = HardClusterAssigner(
            n_vars=self.n_vars,
            num_clusters=args.num_clusters,
            method=args.clustering_method,
            device=self.device
        )
        
        # Initialize empty ModuleList for cluster models
        self.cluster_models = nn.ModuleList()
        self.cluster_sizes = {}
        
        # Move everything to device
        self.to(self.device)
        
    def initialize_clusters(self, full_data):
        """Initialize clusters using full training data"""
        full_data = full_data.to(self.device)
        # Force cluster update using full data
        cluster_assignments = self.cluster_assigner(full_data, if_update=True)
        
        # Create cluster models based on assignments
        self.cluster_models = nn.ModuleList()
        self.cluster_sizes = {}
        
        for cluster_idx in range(self.num_clusters):
            cluster_mask = (cluster_assignments == cluster_idx)
            if cluster_mask.any():
                cluster_channels = torch.where(cluster_mask)[0]
                num_channels = len(cluster_channels)
                self.cluster_sizes[cluster_idx] = num_channels
                
                # Create cluster-specific model with correct number of channels
                self.cluster_models.append(
                    TSMixerBlock(
                        in_len=self.in_len,
                        out_len=self.out_len,
                        d_ff=self.d_ff,
                        n_layers=self.args.n_layers,
                        enc_in=num_channels,  # Use actual number of channels in cluster
                        dropout=self.args.dropout
                    ).to(self.device)
                )
        
        print(f"Initial cluster assignments: {cluster_assignments.cpu().numpy()}")
        print(f"Cluster sizes: {self.cluster_sizes}")
        
        return cluster_assignments
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.to(self.device)
        
        # Apply RevIN normalization
        x = self.rev_in(x, 'norm')
        
        # Get cluster assignments
        cluster_assignments = self.cluster_assigner.cluster_assignments
        
        # Initialize output tensor with correct shape [batch_size, out_len, enc_in]
        outputs = torch.zeros(batch_size, self.out_len, self.enc_in).to(self.device)
        
        # Process each cluster
        model_idx = 0
        for cluster_idx in range(self.num_clusters):
            cluster_mask = (cluster_assignments == cluster_idx)
            if not cluster_mask.any():
                continue
                
            # Select data for current cluster
            cluster_channels = torch.where(cluster_mask)[0]
            cluster_input = x[:, :, cluster_channels]
            
            # Process with corresponding model
            cluster_output = self.cluster_models[model_idx](cluster_input)
            model_idx += 1
            
            # Place outputs back in correct positions
            outputs[:, :, cluster_channels] = cluster_output
        
        # Apply inverse normalization
        outputs = self.rev_in(outputs, 'denorm')
        return outputs
    
    def get_current_assignments(self):
        """Get current cluster assignments"""
        return self.cluster_assigner.cluster_assignments.cpu().numpy()
    
    def get_clustering_metrics(self):
        """Get clustering metrics"""
        return self.cluster_assigner.get_metrics() 