import torch
import torch.nn as nn
from models.hcm1.preprocessor import HardClusterAssigner
from models.Rev_in import RevIN
from models.hcm1.blocks import TSMixerBlock

class ClusterTSMixer(nn.Module):
    """TSMixer model adapted for cluster-specific processing"""
    def __init__(self, num_features, args):
        super().__init__()
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.model = TSMixerBlock(
            in_len=args.seq_len,
            out_len=args.pred_len,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            enc_in=num_features,
            dropout=args.dropout
        ).to(self.device)
        
    def forward(self, x):
        return self.model(x)

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
        self.args = args
        
        # Normalization layer for full input
        self.rev_in = RevIN(num_features=args.enc_in).to(self.device)
        
        # Clustering module
        self.cluster_assigner = HardClusterAssigner(
            n_vars=self.n_vars,
            num_clusters=args.num_clusters,
            method=args.clustering_method,
            device=self.device
        ).to(self.device)
        
        # Initialize cluster models list
        self.cluster_models = nn.ModuleList().to(self.device)
        self.cluster_sizes = {}
        
    def initialize_clusters(self, x):
        """Initialize clusters and models with fixed assignments"""
        x = x.to(self.device)
        # Force cluster update
        cluster_assignments = self.cluster_assigner(x, if_update=True)
        
        # Create new cluster models
        self.cluster_models = nn.ModuleList()
        self.cluster_sizes = {}
        
        for cluster_idx in range(self.num_clusters):
            cluster_mask = (cluster_assignments == cluster_idx)
            if not cluster_mask.any():
                continue
            cluster_channels = torch.where(cluster_mask)[0]
            num_channels = len(cluster_channels)
            self.cluster_sizes[cluster_idx] = num_channels
            
            # Create cluster-specific model
            self.cluster_models.append(ClusterTSMixer(num_channels, self.args))
    
    def load_state_dict_with_init(self, state_dict, x):
        """Load state dict after initializing clusters with given data"""
        try:
            # First initialize clusters
            self.initialize_clusters(x)
            # Now try to load the state dict
            try:
                self.load_state_dict(state_dict)
                print("Successfully loaded state dict with matching cluster sizes")
            except:
                print("Warning: Could not load previous state dict. Starting with fresh weights.")
                # Don't raise the exception - we'll continue with fresh weights
                pass
        except Exception as e:
            print(f"Error during cluster initialization: {str(e)}")
            raise e
    
    def forward(self, x, if_update=False):
        batch_size = x.shape[0]
        x = x.to(self.device)
        
        # Initialize clusters if not done yet
        if len(self.cluster_models) == 0:
            self.initialize_clusters(x)
        
        # Apply RevIN normalization
        x = self.rev_in(x, 'norm')
        
        # Get cluster assignments (without updating if not requested)
        cluster_assignments = self.cluster_assigner(x, if_update)
        
        # Initialize output tensor
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