import torch
import torch.nn as nn
from .layers import HardClusterAssigner
from .patch_layer import Cluster_wise_TSMixer

class MLPTime(nn.Module):
    def __init__(self, seq_len, dropout_rate):
        super(MLPTime, self).__init__()
        self.fc = nn.Linear(seq_len, seq_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MLPFeat(nn.Module):
    def __init__(self, C, ff_dim, dropout_rate=0.1):
        super(MLPFeat, self).__init__()
        self.fc1 = nn.Linear(C, ff_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, C)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class MixerLayer(nn.Module):
    def __init__(self, n_vars, seq_len, ff_dim, dropout):
        super(MixerLayer, self).__init__()
        self.mlp_time = MLPTime(seq_len, dropout)
        self.mlp_feat = MLPFeat(n_vars, ff_dim, dropout)
        
    def batch_norm_2d(self, x):
        return (x - x.mean()) / x.std()
    
    def forward(self, x):
        res_x = x
        x = self.batch_norm_2d(x)
        x = x.transpose(1, 2)
        x = self.mlp_time(x)
        x = x.transpose(1, 2) + res_x
        res_x = x
        x = self.batch_norm_2d(x)
        x = self.mlp_feat(x) + res_x
        return x

class HardClusterTSMixer(nn.Module):
    def __init__(self, args):
        super(HardClusterTSMixer, self).__init__()
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.n_cluster = args.n_cluster
        self.d_model = args.d_model
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        
        # Regular mixer layers
        self.mixer_layers = nn.ModuleList([
            MixerLayer(
                n_vars=self.n_vars,
                seq_len=self.in_len,
                ff_dim=args.d_ff,
                dropout=args.dropout
            ) for _ in range(args.n_layers)
        ])
        
        # Cluster assignment module
        self.cluster_assigner = HardClusterAssigner(
            n_vars=self.n_vars,
            n_cluster=self.n_cluster,
            seq_len=self.in_len,
            d_model=self.d_model,
            device=self.device
        )
        
        # Cluster-specific TSMixer projections
        self.cluster_projector = Cluster_wise_TSMixer(
            n_cluster=self.n_cluster,
            n_vars=self.n_vars,
            in_len=self.in_len,
            out_len=self.out_len,
            hidden_size=args.d_ff,
            dropout=args.dropout,
            device=self.device
        )

    def forward(self, x, return_clusters=False):
        # Regular mixer layers
        for layer in self.mixer_layers:
            x = layer(x)
        
        # Get cluster assignments
        assignments = self.cluster_assigner(x)
        
        # Apply cluster-specific TSMixer projections
        x = x.transpose(1, 2)
        outputs = self.cluster_projector(x, assignments)
        outputs = outputs.transpose(1, 2)
        
        if return_clusters:
            return outputs, assignments
        return outputs