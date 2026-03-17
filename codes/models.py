import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SpectralECFDetector(nn.Module):
    def __init__(self, in_channels=2, M=128):
        """
        in_channels: The dimension of your raw data 'd' (e.g., 2 or 50).
        M: The number of learnable frequencies for the ECF projection.
        """
        super().__init__()
        
        # 1. Learnable Frequency Matrix (replaces the CNN)
        # Shape: (M frequencies, in_channels)
        # We initialize with standard normal, but this will update during training
        self.U = nn.Parameter(torch.randn(M, in_channels))
        
        # 2. Multiscale ECF Logic
        # We keep the scales to give the model structural flexibility 
        # across different frequency bandwidths.
        self.scales = nn.Parameter(torch.tensor([0.1, 1.0, 2.0]))
        
    def get_fingerprint(self, x):
        """
        Takes a time window and returns its L2-normalized ECF fingerprint.
        x shape: (Batch, Window_Length, Channels/Dimensions)
        """
        # x is already (Batch, L, Channels), no need to transpose for a CNN anymore
        z_list = []
        
        for s in self.scales:
            # Scale the learnable frequencies
            U_scaled = self.U * s 
            
            # Project raw data onto frequencies
            # x shape: (Batch, L, Channels)
            # U_scaled.T shape: (Channels, M)
            # S shape: (Batch, L, M)
            S = torch.matmul(x, U_scaled.T)
            
            # For a true ECF, apply trig functions THEN average over the time window (L)
            # This computes the empirical expectation of the characteristic function
            cos_S = torch.mean(torch.cos(S), dim=1) # Shape: (Batch, M)
            sin_S = torch.mean(torch.sin(S), dim=1) # Shape: (Batch, M)
            
            z_list.append(torch.cat([cos_S, sin_S], dim=-1))
        
        # Concatenate all scales and normalize to the unit hypersphere
        z = torch.cat(z_list, dim=-1)
        return F.normalize(z, p=2, dim=-1)

    def forward(self, anchor, positive, negative):
        """Forward pass for Triplet Contrastive Training."""
        z_a = self.get_fingerprint(anchor)
        z_p = self.get_fingerprint(positive)
        z_n = self.get_fingerprint(negative)
        return z_a, z_p, z_n