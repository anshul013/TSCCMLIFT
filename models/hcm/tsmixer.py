import torch
import torch.nn as nn
from models.Rev_in import RevIN
from .layers import HardClusterAssigner

class MlpBlockTimesteps(nn.Module):
    """MLP for timesteps with 1 layer"""
    def __init__(self, seq_len, dropout_factor, activation):
        super(MlpBlockTimesteps, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(seq_len)
        self.linear_layer = nn.Linear(seq_len, seq_len)
        if activation=="gelu":
            self.activation_layer = nn.GELU()
        elif activation=="relu":
            self.activation_layer = nn.ReLU()
        else:
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x):
        y = self.normalization_layer(x)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer(y)
        y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 2)
        return x + y

class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, mlp_dim, dropout_factor, activation, single_layer_mixer):
        super(MlpBlockFeatures, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(channels)
        self.single_layer_mixer = single_layer_mixer
        if self.single_layer_mixer:
            self.linear_layer1 = nn.Linear(channels, channels)
        else:
            self.linear_layer1 = nn.Linear(channels, mlp_dim)
            self.linear_layer2 = nn.Linear(mlp_dim, channels)
        if activation=="gelu":
            self.activation_layer = nn.GELU()
        elif activation=="relu":
            self.activation_layer = nn.ReLU()
        else:
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x):
        y = torch.swapaxes(x, 1, 2)
        y = self.normalization_layer(y)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer1(y)
        if self.activation_layer is not None:
            y = self.activation_layer(y)
        if not(self.single_layer_mixer):
            y = self.dropout_layer(y)
            y = self.linear_layer2(y)
        y = self.dropout_layer(y)
        return x + y

class MixerBlock(nn.Module):
    """Mixer block layer"""
    def __init__(self, channels, hidden_size, seq_len, dropout_factor, activation, single_layer_mixer):
        super(MixerBlock, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.timesteps_mixer = MlpBlockTimesteps(seq_len, dropout_factor, activation)
        self.channels_mixer = MlpBlockFeatures(channels, hidden_size, dropout_factor, activation, single_layer_mixer)
    
    def forward(self, x):
        y = self.timesteps_mixer(x)   
        y = self.channels_mixer(y)
        return y

class ClusterTSMixer(nn.Module):
    """TSMixer for cluster-specific projection"""
    def __init__(self, seq_len, pred_len, hidden_size, dropout_factor, activation, single_layer_mixer, num_blocks=2):
        super().__init__()
        self.num_blocks = num_blocks
        self.pred_len = pred_len
        
        # TSMixer blocks for processing
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                channels=seq_len,  # Note: we operate on transposed input
                hidden_size=hidden_size,
                seq_len=seq_len,
                dropout_factor=dropout_factor,
                activation=activation,
                single_layer_mixer=single_layer_mixer
            ) for _ in range(num_blocks)
        ])
        
        # Final projection to prediction length
        self.output_proj = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        # x: [bs, vars_in_cluster, seq_len]
        x = torch.swapaxes(x, 1, 2)  # [bs, seq_len, vars_in_cluster]
        
        # Apply TSMixer blocks
        for block in self.mixer_blocks:
            x = block(x)
            
        # Project to prediction length
        x = self.output_proj(x)  # [bs, pred_len, vars_in_cluster]
        x = torch.swapaxes(x, 1, 2)  # [bs, vars_in_cluster, pred_len]
        return x

class HardClusterTSMixer(nn.Module):
    def __init__(self, configs):
        super(HardClusterTSMixer, self).__init__()
        self.num_blocks = configs.num_blocks
        self.channels = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_cluster = configs.n_cluster
        
        # RevIN normalization
        self.rev_norm = RevIN(self.channels, affine=configs.affine)
        
        # Regular mixer blocks
        self.mixer_block = MixerBlock(
            channels=self.channels,
            features_block_mlp_dims=configs.hidden_size,
            seq_len=self.seq_len,
            dropout_factor=configs.dropout,
            activation=configs.activation,
            single_layer_mixer=configs.single_layer_mixer
        )
        
        # Cluster assignment module
        self.cluster_assigner = HardClusterAssigner(
            n_vars=self.channels,
            n_cluster=self.n_cluster,
            seq_len=self.seq_len,
            hidden_size=configs.hidden_size
        )
        
        # Cluster-specific TSMixers
        self.cluster_projectors = nn.ModuleList([
            ClusterTSMixer(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_size=configs.hidden_size,
                dropout_factor=configs.dropout,
                activation=configs.activation,
                single_layer_mixer=configs.single_layer_mixer
            ) for _ in range(self.n_cluster)
        ])

    def forward(self, x, return_clusters=False):
        # x: [Batch, Input length, Channel]
        x = self.rev_norm(x, 'norm')
        
        # Regular mixer blocks
        for _ in range(self.num_blocks):
            x = self.mixer_block(x)
            
        # Get cluster assignments
        assignments = self.cluster_assigner(x)  # [n_vars, n_cluster]
        
        # Apply cluster-specific TSMixers
        x = torch.swapaxes(x, 1, 2)  # [Batch, Channel, Length]
        output = torch.zeros(x.size(0), self.channels, self.pred_len).to(x.device)
        
        for i in range(self.n_cluster):
            cluster_mask = assignments[:, i]
            if cluster_mask.sum() == 0:
                continue
            
            cluster_vars = x[:, cluster_mask.bool(), :]
            projected = self.cluster_projectors[i](cluster_vars)
            output[:, cluster_mask.bool(), :] = projected
            
        output = torch.swapaxes(output, 1, 2)  # [Batch, pred_len, Channel]
        output = self.rev_norm(output, 'denorm')
        
        if return_clusters:
            return output, assignments
        return output