import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hcm.layers import *
from models.Rev_in import RevIN
from models.hcm.blocks import *

class TSMixerH(nn.Module):
    """Hard Clustering variant of TSMixer"""
    def __init__(self, args):
        super(TSMixerH, self).__init__()
        # Basic parameters
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.num_clusters = args.num_clusters
        self.d_ff = args.d_ff
        self.d_model = args.d_model
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.enc_in = args.enc_in
        # Normalization layer
        self.rev_in = RevIN(num_features=args.enc_in)
        
        # Clustering module
        self.cluster_assigner = ClusterAssigner(
            n_vars=self.n_vars,
            num_clusters=self.num_clusters,
            method=args.clustering_method,
            device=self.device
        )
        
        # Create separate TSMixer models for each cluster
        self.cluster_models = nn.ModuleList([
            TSMixerBlock(
                in_len=self.in_len,
                out_len=self.out_len,
                d_model=self.d_model,
                d_ff=self.d_ff,
                n_layers=args.n_layers,
                enc_in=args.enc_in,
                dropout=args.dropout
            ) for _ in range(self.num_clusters)
        ])
        
        # Track current assignments
        self.current_assignments = None

    def forward(self, x, if_update=False):
        """
        Args:
            x: Input tensor of shape [batch_size, n_vars, seq_len]
            if_update: Boolean indicating whether to update cluster assignments
        Returns:
            output: Tensor of shape [batch_size, n_vars, out_len]
        """
        print("Shape of x before RevIN:",x.shape)
        # Apply RevIN normalization
        x = self.rev_in(x, 'norm')
        print("Shape of x after RevIN:",x.shape)
        # Get and store cluster assignments
        cluster_assignments = self.cluster_assigner(x, if_update)
        self.current_assignments = cluster_assignments
        # print("Cluster assignments:",cluster_assignments)
        print("Shape of cluster_assignments:",cluster_assignments.shape)
        # print("Shape of x after cluster_assigner:",x.shape)
        # Initialize output tensor
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, self.enc_in, self.out_len).to(self.device)
        print("Shape of outputs:",outputs.shape)
        x = x.transpose(1, 2)
        print("Shape of x after transpose:",x.shape)
        # Keep track of output position
        current_pos = 0
        # Process each cluster separately
        for cluster_idx in range(self.num_clusters):
            # Get variables belonging to current cluster
            cluster_mask = (cluster_assignments == cluster_idx)
            # print("cluster_mask:",cluster_mask)
            print("Shape of cluster_mask:",cluster_mask.shape)
            cluster_size = cluster_mask.sum().item()
            if cluster_size == 0:
                continue
            # Select data for current cluster
            cluster_x = x[:, :, cluster_mask]
            # print("cluster_x:",cluster_x)
            print("Shape of cluster_x:",cluster_x.shape)
            # Process with corresponding TSMixer
            cluster_output = self.cluster_models[cluster_idx](cluster_x)
            print("Shape of cluster_output:",cluster_output.shape)
            # Place outputs back in correct positions
            outputs[:, :, current_pos:current_pos + cluster_size] = cluster_output
            print("Shape of outputs:",outputs.shape)
            current_pos += cluster_size
        # Apply inverse normalization
        outputs = self.rev_in(outputs, 'denorm')
        print("Shape of outputs:",outputs.shape)
        outputs = outputs.transpose(1, 2)
        print("Shape of outputs after transpose:",outputs.shape)
        return outputs

    def get_current_assignments(self):
        """Returns current cluster assignments for analysis"""
        return self.current_assignments

    def get_clustering_metrics(self):
        """Returns clustering metrics from the assigner"""
        return self.cluster_assigner.get_clustering_metrics() 