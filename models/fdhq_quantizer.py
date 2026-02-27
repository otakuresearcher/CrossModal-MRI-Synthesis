"""
Frequency-Decomposed Hierarchical Quantization (FDHQ).

Novel contribution: Uses frequency-domain hierarchy instead of spatial hierarchy.
Separate codebooks for low-frequency (structures) vs high-frequency (textures).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FrequencyDecomposer(nn.Module):
    """
    Learnable frequency decomposition module.
    
    Separates latent features into low-frequency (global structure)
    and high-frequency (local texture) components.
    """
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        
        # Learnable low-pass filter
        self.low_pass = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()  # Soft gating
        )
        
        # Refinement for high-frequency
        self.high_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU()
        )
        
    def forward(self, z):
        """
        Decompose z into low and high frequency components.
        
        Returns:
            z_low: Low-frequency (smooth, global structure)
            z_high: High-frequency (edges, textures)
        """
        # Learnable low-pass filter
        gate = self.low_pass(z)
        z_low = z * gate
        
        # High frequency = residual
        z_high = z - z_low
        z_high = self.high_refine(z_high)
        
        return z_low, z_high


class FrequencyQuantizer(nn.Module):
    """
    Single quantizer for one frequency band.
    Uses EMA codebook updates for stability.
    """
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        distance: str = 'l2'
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.distance = distance
        
        # Codebook with EMA
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_count', torch.zeros(num_embeddings))
        self.register_buffer('ema_weight', torch.randn(num_embeddings, embedding_dim))
        
        self.embedding.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.ema_weight.data.copy_(self.embedding.data)
        
    def forward(self, z):
        """
        Quantize input tensor using nearest neighbor lookup.
        Uses EMA codebook updates during training.
        """
        # Reshape for quantization
        z_shape = z.shape
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flat = z.view(-1, self.embedding_dim)
        
        # Distance computation
        if self.distance == 'l2':
            distances = (
                torch.sum(z_flat**2, dim=1, keepdim=True) +
                torch.sum(self.embedding**2, dim=1) -
                2 * torch.matmul(z_flat, self.embedding.t())
            )
        else:  # cosine
            norm_z = F.normalize(z_flat, dim=1)
            norm_emb = F.normalize(self.embedding, dim=1)
            distances = 1 - torch.matmul(norm_z, norm_emb.t())
        
        # Find nearest
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding).view(z.shape)
        
        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # EMA update during training
        if self.training:
            with torch.no_grad():
                # Count how many times each code is used
                encodings_sum = encodings.sum(0)
                dw = torch.matmul(encodings.t(), z_flat)
                
                self.ema_count.mul_(self.ema_decay).add_(encodings_sum, alpha=1 - self.ema_decay)
                self.ema_weight.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
                
                # Laplace smoothing
                n = self.ema_count.sum()
                count = (self.ema_count + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                
                self.embedding.data.copy_(self.ema_weight / count.unsqueeze(1))
        
        # Straight-through
        quantized = z + (quantized - z).detach()
        quantized = rearrange(quantized, 'b h w c -> b c h w').contiguous()
        
        # Perplexity
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices


class FDHQQuantizer(nn.Module):
    """
    Frequency-Decomposed Hierarchical Quantization.
    
    Novel contribution: Instead of using spatial hierarchy (like VQ-VAE-2),
    we decompose features into frequency bands:
    - Low-frequency: Global structure, tissue boundaries
    - High-frequency: Fine textures, edges, details
    
    Each frequency band has its own codebook, allowing:
    1. Better representation of medical image characteristics
    2. Separate optimization of structure vs texture
    3. More interpretable latent codes
    """
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings_low: int = 512,
        num_embeddings_high: int = 512,
        commitment_cost: float = 0.25,
        low_weight: float = 1.0,    # Weight for low-freq reconstruction
        high_weight: float = 1.0,   # Weight for high-freq reconstruction
        ema_decay: float = 0.99
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.low_weight = low_weight
        self.high_weight = high_weight
        
        # Frequency decomposition
        self.decomposer = FrequencyDecomposer(embedding_dim)
        
        # Separate quantizers for each frequency band
        self.quantizer_low = FrequencyQuantizer(
            num_embeddings_low, embedding_dim, 
            commitment_cost=commitment_cost,
            ema_decay=ema_decay
        )
        self.quantizer_high = FrequencyQuantizer(
            num_embeddings_high, embedding_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay
        )
        
        # Fusion layer to combine quantized components
        self.fusion = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, embedding_dim, 1),
            nn.GroupNorm(8, embedding_dim),
            nn.GELU(),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1)
        )
        
    def forward(self, z):
        """
        Forward pass with frequency-decomposed quantization.
        
        Args:
            z: Input latent tensor (B, C, H, W)
            
        Returns:
            quantized: Reconstructed quantized tensor
            loss: Combined VQ loss
            info: Dictionary with per-band metrics
        """
        # Decompose into frequency bands
        z_low, z_high = self.decomposer(z)
        
        # Quantize each band separately
        q_low, loss_low, perp_low, idx_low = self.quantizer_low(z_low)
        q_high, loss_high, perp_high, idx_high = self.quantizer_high(z_high)
        
        # Fuse quantized representations
        combined = torch.cat([q_low, q_high], dim=1)
        quantized = self.fusion(combined)
        
        # Add residual from low-frequency (most important structure)
        quantized = quantized + q_low
        
        # Combined loss
        loss = self.low_weight * loss_low + self.high_weight * loss_high
        
        # Info for logging
        info = {
            'perplexity_low': perp_low,
            'perplexity_high': perp_high,
            'perplexity': (perp_low + perp_high) / 2,
            'indices_low': idx_low,
            'indices_high': idx_high,
            'z_low': z_low,
            'z_high': z_high
        }
        
        return quantized, loss, info
    
    def get_codebook_usage(self):
        """Return codebook utilization statistics."""
        with torch.no_grad():
            low_count = (self.quantizer_low.ema_count > 0.01).sum().item()
            high_count = (self.quantizer_high.ema_count > 0.01).sum().item()
            
        return {
            'low_active': low_count,
            'low_total': self.quantizer_low.num_embeddings,
            'high_active': high_count,
            'high_total': self.quantizer_high.num_embeddings
        }
