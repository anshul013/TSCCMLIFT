import torch
import torch.nn as nn
import torch.nn.functional as F

class TSMixerBlock(nn.Module):
    """TSMixer block for processing clustered variables"""
    def __init__(self, in_len, out_len, d_model, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        
        # Mixer layers
        self.mixer_layers = nn.ModuleList([
            MixerLayer(
                seq_len=in_len,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Output projection
        self.projection = nn.Linear(in_len, out_len)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, cluster_size, seq_len]
        Returns:
            output: Tensor of shape [batch_size, cluster_size, out_len]
        """
        # Process through mixer layers
        for layer in self.mixer_layers:
            x = layer(x)
        
        # Project to output length
        x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        return x

class MixerLayer(nn.Module):
    """Individual mixer layer"""
    def __init__(self, seq_len, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(seq_len)
        self.norm2 = nn.LayerNorm(seq_len)
        
        # Temporal mixing
        self.mlp1 = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len)
        )
        
        # Channel mixing
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        # Temporal mixing
        y = self.norm1(x.transpose(1, 2))
        y = self.mlp1(y)
        x = x + y.transpose(1, 2)
        
        # Channel mixing
        y = self.norm2(x)
        y = self.mlp2(y)
        x = x + y
        
        return x 