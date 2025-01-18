import torch
import torch.nn as nn
from models.hcm1.preprocessor import HardClusterAssigner
from models.Rev_in import RevIN

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
        self.d_model = args.d_model
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.enc_in = args.enc_in
        
        # Normalization layer
        self.rev_in = RevIN(num_features=args.enc_in)
        
        # Clustering module
        self.cluster_assigner = HardClusterAssigner(
            n_vars=self.n_vars,
            num_clusters=args.num_clusters,
            method=args.clustering_method,
            device=self.device
        )
        
        # Create separate TSMixer models for each cluster
        from models.TSMixer import Model as TSMixer
        self.cluster_models = nn.ModuleList([
            TSMixer(args) for _ in range(self.num_clusters)
        ])
        
    def forward(self, x, if_update=False):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_vars]
            if_update: Whether to update cluster assignments
        Returns:
            outputs: Output tensor of shape [batch_size, pred_len, n_vars]
        """
        batch_size = x.shape[0]
        
        # Apply RevIN normalization
        x = self.rev_in(x, 'norm')
        
        # Get cluster assignments
        cluster_assignments = self.cluster_assigner(x, if_update)
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, self.out_len, self.enc_in).to(self.device)
        
        # Process each cluster separately
        for cluster_idx in range(self.num_clusters):
            cluster_mask = (cluster_assignments == cluster_idx)
            if not cluster_mask.any():
                continue
                
            # Select data for current cluster
            cluster_channels = torch.where(cluster_mask)[0]
            cluster_input = x[:, :, cluster_channels]
            
            # Process with corresponding model
            cluster_output = self.cluster_models[cluster_idx](cluster_input)
            
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