import torch
import torch.nn as nn

class TSMixerBlock(nn.Module):
    """TSMixer block for processing clustered variables"""
    def __init__(self, in_len, out_len, d_model, d_ff, n_layers, enc_in, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        
        # Create mixer block for each layer
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                features_block_mlp_dims=d_ff,
                dropout_factor=dropout,
                activation='relu',
                single_layer_mixer=False
            ) for _ in range(n_layers)
        ])
        
        
        
    def forward(self, x):
        # Process through mixer blocks
        for block in self.mixer_blocks:
            x = block(x)
        print("Shape of x after mixer blocks:",x.shape)
        
        # Project to output length
        in_len = x.size(1)
        out_len = x.size(2)
        self.output_linear = nn.Linear(in_len, out_len).to(x.device)
        x = torch.swapaxes(x, 1, 2)
        x = self.output_linear(x)
        x = torch.swapaxes(x, 1, 2)
        return x

class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, mlp_dim, dropout_factor, activation, single_layer_mixer):
        super().__init__()
        self.single_layer_mixer = single_layer_mixer
        self.mlp_dim = mlp_dim
        self.dropout_factor = dropout_factor
        
        if activation == "gelu":
            self.activation_layer = nn.GELU()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        else:
            self.activation_layer = None
            
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x):
        # Create layers dynamically based on input size
        channels = x.size(2)  # Get number of channels
        
        self.normalization_layer = nn.BatchNorm1d(channels).to(x.device)
        

        if self.single_layer_mixer:
            self.linear_layer1 = nn.Linear(channels, channels).to(x.device)
        else:
            self.linear_layer1 = nn.Linear(channels, self.mlp_dim).to(x.device)
            self.linear_layer2 = nn.Linear(self.mlp_dim, channels).to(x.device)
        
        # Forward pass
        y = torch.swapaxes(x, 1, 2)  # [batch_size, channels, seq_len]
        print("Shape of y before normalization:",y.shape)
        y = self.normalization_layer(y)
        print("Shape of y after normalization:",y.shape)
        y = torch.swapaxes(y, 1, 2)  # [batch_size, seq_len, channels]
        y = self.linear_layer1(y)
        print("Shape of y after linear layer 1:",y.shape)
        
        if self.activation_layer is not None:
            y = self.activation_layer(y)
            
        if not self.single_layer_mixer:
            y = self.dropout_layer(y)
            y = self.linear_layer2(y)
            
        y = self.dropout_layer(y)
        print("Shape of y after dropout:",y.shape)
        return x + y

class MlpBlockTimesteps(nn.Module):
    """MLP for timesteps with 1 layer"""
    def __init__(self, dropout_factor, activation):
        super().__init__()
        self.dropout_factor = dropout_factor
        
        if activation == "gelu":
            self.activation_layer = nn.GELU()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        else:
            self.activation_layer = None
            
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x):
        # Create layers dynamically based on input size
        seq_len = x.size(1)  # Get sequence length
        self.normalization_layer = nn.BatchNorm1d(seq_len).to(x.device)
        self.linear_layer = nn.Linear(seq_len, seq_len).to(x.device)
        print("Shape of x before normalization:",x.shape)
        # Forward pass
        y = self.normalization_layer(x)
        print("Shape of y after normalization:",y.shape)
        y = torch.swapaxes(y, 1, 2)
        print("Shape of y after swapaxes:",y.shape)
        y = self.linear_layer(y)
        print("Shape of y after linear layer:",y.shape)
        y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 2)
        print("Shape of y after swapaxes:",y.shape)
        return x + y

class MixerBlock(nn.Module):
    """Mixer block layer"""
    def __init__(self, features_block_mlp_dims, dropout_factor, activation, single_layer_mixer):
        super().__init__()
        
        # Timesteps mixing block
        self.timesteps_mixer = MlpBlockTimesteps(dropout_factor, activation)
        # Features mixing block
        self.channels_mixer = MlpBlockFeatures(features_block_mlp_dims, dropout_factor, activation, single_layer_mixer)
    
    def forward(self, x):
        y = self.timesteps_mixer(x)
        y = self.channels_mixer(y)
        return y 