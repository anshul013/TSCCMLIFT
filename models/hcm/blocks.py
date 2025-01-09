import torch
import torch.nn as nn

class TSMixerBlock(nn.Module):
    """TSMixer block for processing clustered variables"""
    def __init__(self, in_len, out_len, d_model, d_ff, n_layers, enc_in, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.in_len = in_len
        self.out_len = out_len
        
        # Create mixer block for each layer
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                channels=enc_in,
                features_block_mlp_dims=d_ff,
                seq_len=in_len,
                dropout_factor=dropout,
                activation='relu',
                single_layer_mixer=False
            ) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_linear = nn.Linear(in_len, out_len)
        
    def forward(self, x):
        # Process through mixer blocks
        for block in self.mixer_blocks:
            x = block(x)
            
        # Project to output length
        x = torch.swapaxes(x, 1, 2)
        x = self.output_linear(x)
        x = torch.swapaxes(x, 1, 2)
        return x

class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, mlp_dim, dropout_factor, activation, single_layer_mixer):
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
        print(x.shape)
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