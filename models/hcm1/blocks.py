import torch
import torch.nn as nn
from models.Rev_in import RevIN

class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, mlp_dim, dropout_factor, activation, single_layer_mixer=False):
        super().__init__()
        self.normalization_layer = nn.BatchNorm1d(channels)
        self.single_layer_mixer = single_layer_mixer
        if self.single_layer_mixer:
            self.linear_layer1 = nn.Linear(channels, channels)
        else:
            self.linear_layer1 = nn.Linear(channels, mlp_dim)
            self.linear_layer2 = nn.Linear(mlp_dim, channels)
            
        if activation == "gelu":
            self.activation_layer = nn.GELU()
        elif activation == "relu":
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
        if not self.single_layer_mixer:
            y = self.dropout_layer(y)
            y = self.linear_layer2(y)
        y = self.dropout_layer(y)
        return x + y

class MlpBlockTimesteps(nn.Module):
    """MLP for timesteps with 1 layer"""
    def __init__(self, seq_len, dropout_factor, activation):
        super().__init__()
        self.normalization_layer = nn.BatchNorm1d(seq_len)
        self.linear_layer = nn.Linear(seq_len, seq_len)
        if activation == "gelu":
            self.activation_layer = nn.GELU()
        elif activation == "relu":
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

class MixerBlock(nn.Module):
    """Mixer block layer"""
    def __init__(self, channels, features_block_mlp_dims, seq_len, dropout_factor, activation, single_layer_mixer):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        # Timesteps mixing block
        self.timesteps_mixer = MlpBlockTimesteps(seq_len, dropout_factor, activation)
        # Features mixing block
        self.channels_mixer = MlpBlockFeatures(channels, features_block_mlp_dims, dropout_factor, activation, single_layer_mixer)

    def forward(self, x):
        y = self.timesteps_mixer(x)
        y = self.channels_mixer(y)
        return y

class TSMixerBlock(nn.Module):
    """TSMixer block for processing clustered variables"""
    def __init__(self, in_len, out_len, d_model, d_ff, n_layers, enc_in, dropout=0.1):
        super().__init__()
        self.num_blocks = n_layers
        self.channels = enc_in
        self.pred_len = out_len
        
        # Create mixer block
        self.mixer_block = MixerBlock(
            channels=enc_in,
            features_block_mlp_dims=d_ff,
            seq_len=in_len,
            dropout_factor=dropout,
            activation='relu',
            single_layer_mixer=False
        )
        
        # RevIN normalization
        self.rev_norm = RevIN(num_features=enc_in)
        
        # Output projection
        self.output_linear_layers = nn.ModuleList([
            nn.Linear(in_len, out_len) for _ in range(enc_in)
        ])

    def to(self, device):
        """Ensures all submodules are on the correct device"""
        super().to(device)
        self.mixer_block = self.mixer_block.to(device)
        self.rev_norm = self.rev_norm.to(device)
        self.output_linear_layers = self.output_linear_layers.to(device)
        return self

    def forward(self, x):
        device = x.device
        # Ensure all components are on the same device
        self.to(device)
        
        # Apply RevIN normalization
        x = self.rev_norm(x, 'norm')
        
        # Apply mixer blocks
        for _ in range(self.num_blocks):
            x = self.mixer_block(x)
            
        # Project to prediction length
        x = torch.swapaxes(x, 1, 2)
        y = torch.zeros([x.size(0), x.size(1), self.pred_len], dtype=x.dtype, device=device)
        
        for c in range(self.channels):
            y[:, c, :] = self.output_linear_layers[c](x[:, c, :].clone())
            
        y = torch.swapaxes(y, 1, 2)
        y = self.rev_norm(y, 'denorm')
        return y 