"""
Enhanced VQ-VAE2 Model.

Combines all novel contributions:
1. EWSCS - Error-Weighted Semantic Coreset Selection
2. CMCR - Cross-Modal Consistency Regularization  
3. FDHQ - Frequency-Decomposed Hierarchical Quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ewscs_quantizer import EWSCSQuantizer
from .fdhq_quantizer import FDHQQuantizer
from .cmcr_module import CMCRModule


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm for stability."""
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels)
        )
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.activation(x + self.block(x))


class SelfAttention(nn.Module):
    """Self-attention module for capturing long-range dependencies."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Normalize and reshape
        x_norm = self.norm(x)
        x_flat = rearrange(x_norm, 'b c h w -> b (h w) c')
        
        # Self-attention
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        
        # Reshape back and residual
        attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x + attn_out


class EnhancedEncoder(nn.Module):
    """
    Enhanced hierarchical encoder with attention.
    
    Features:
    - 3-level hierarchy for multi-scale features
    - Self-attention at bottleneck
    - Skip connections for decoder
    """
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super().__init__()
        
        # Level 1: 256 -> 128
        self.enc_1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            ResidualBlock(hidden_channels)
        )
        
        # Level 2: 128 -> 64
        self.enc_2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            ResidualBlock(hidden_channels)
        )
        
        # Level 3: 64 -> 32 (optional deeper encoding)
        self.enc_3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.GELU(),
            ResidualBlock(hidden_channels * 2),
            SelfAttention(hidden_channels * 2)  # Attention at bottleneck
        )
        
        # Project to latent space
        self.to_latent_top = nn.Sequential(
            ResidualBlock(hidden_channels * 2),
            nn.Conv2d(hidden_channels * 2, latent_channels, 3, padding=1)
        )
        
        self.to_latent_bottom = nn.Sequential(
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        )
        
    def forward(self, x):
        """
        Returns:
            z_top: Latent at 32x32 (global features)
            z_bottom: Latent at 64x64 (local features)
            skip_128: Skip connection at 128x128
        """
        # Encode down
        h1 = self.enc_1(x)       # 128x128
        h2 = self.enc_2(h1)      # 64x64
        h3 = self.enc_3(h2)      # 32x32
        
        # Project to latent
        z_top = self.to_latent_top(h3)        # 32x32
        z_bottom = self.to_latent_bottom(h2)  # 64x64
        
        return z_top, z_bottom, h1  # Return skip for decoder


class EnhancedDecoder(nn.Module):
    """
    Enhanced decoder with skip connections and multi-scale fusion.
    """
    def __init__(self, latent_channels, hidden_channels, out_channels):
        super().__init__()
        
        # Upsample top latent: 32 -> 64
        self.upsample_top = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            nn.GroupNorm(8, latent_channels),
            nn.GELU()
        )
        
        # Fuse top and bottom latents
        self.fuse_latents = nn.Sequential(
            nn.Conv2d(latent_channels * 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            ResidualBlock(hidden_channels)
        )
        
        # Decode 64 -> 128
        self.dec_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            ResidualBlock(hidden_channels)
        )
        
        # Skip connection fusion
        self.skip_fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU()
        )
        
        # Decode 128 -> 256
        self.dec_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, out_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z_top, z_bottom, skip_128=None):
        """
        Decode from quantized latents.
        
        Args:
            z_top: Quantized top latent (32x32 or 64x64 after upsampling)
            z_bottom: Quantized bottom latent (64x64)
            skip_128: Optional skip connection from encoder
        """
        # Upsample top to match bottom
        z_top_up = self.upsample_top(z_top)  # 64x64
        
        # Fuse latents
        fused = torch.cat([z_top_up, z_bottom], dim=1)
        h = self.fuse_latents(fused)  # 64x64
        
        # Decode to 128
        h = self.dec_1(h)  # 128x128
        
        # Fuse with skip connection if available
        if skip_128 is not None:
            h = self.skip_fuse(torch.cat([h, skip_128], dim=1))
        
        # Decode to output
        out = self.dec_2(h)  # 256x256
        
        return out


class EnhancedCSVQVAE(nn.Module):
    """
    Enhanced CS-VQ-VAE with all novel contributions.
    
    Novel Features:
    1. EWSCS: Error-weighted codebook re-initialization
    2. CMCR: Cross-modal consistency regularization
    3. FDHQ: Frequency-decomposed quantization
    
    Architecture:
    - Enhanced encoder with attention
    - Hierarchical quantization (top + bottom)
    - FDHQ at bottom level for texture details
    - EWSCS at top level for semantic features
    - Skip connections in decoder
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        latent_channels: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        # EWSCS parameters
        error_weight: float = 0.7,
        num_semantic_clusters: int = 16,
        # FDHQ parameters
        use_fdhq: bool = True,
        low_freq_weight: float = 1.0,
        high_freq_weight: float = 1.0,
        # CMCR parameters
        use_cmcr: bool = True,
        cmcr_weight: float = 0.1
    ):
        super().__init__()
        
        self.use_fdhq = use_fdhq
        self.use_cmcr = use_cmcr
        self.cmcr_weight = cmcr_weight
        
        # Encoder
        self.encoder = EnhancedEncoder(in_channels, hidden_channels, latent_channels)
        
        # Quantizers
        # Top level: EWSCS for semantic/global features
        self.quantizer_top = EWSCSQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_channels,
            commitment_cost=commitment_cost,
            error_weight=error_weight,
            num_semantic_clusters=num_semantic_clusters
        )
        
        # Bottom level: FDHQ or EWSCS
        if use_fdhq:
            self.quantizer_bottom = FDHQQuantizer(
                embedding_dim=latent_channels,
                num_embeddings_low=num_embeddings,
                num_embeddings_high=num_embeddings,
                commitment_cost=commitment_cost,
                low_weight=low_freq_weight,
                high_weight=high_freq_weight
            )
        else:
            self.quantizer_bottom = EWSCSQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=latent_channels,
                commitment_cost=commitment_cost,
                error_weight=error_weight,
                num_semantic_clusters=num_semantic_clusters
            )
        
        # Decoder
        self.decoder = EnhancedDecoder(latent_channels, hidden_channels, in_channels)
        
        # CMCR module (optional)
        if use_cmcr:
            self.cmcr_module = CMCRModule(
                encoder=self.encoder,
                encoder_mode='shared',
                feature_weight=1.0,
                distribution_weight=0.5,
                structure_weight=0.5
            )
        
        # Store for metrics
        self.num_embeddings = num_embeddings
        
    def forward(self, x_source, x_target=None):
        """
        Forward pass.
        
        Args:
            x_source: Source images (post-contrast)
            x_target: Target images (pre-contrast) for CMCR, optional
            
        Returns:
            recon: Reconstructed image
            losses: Dictionary of losses
            metrics: Dictionary of metrics
        """
        # Encode
        z_top, z_bottom, skip_128 = self.encoder(x_source)
        
        # Quantize top
        q_top, vq_loss_top, info_top = self.quantizer_top(z_top)
        perplexity_top, encodings_top, indices_top, alive_top, error_top = info_top
        
        # Quantize bottom
        if self.use_fdhq:
            q_bottom, vq_loss_bottom, info_bottom = self.quantizer_bottom(z_bottom)
            perplexity_bottom = info_bottom['perplexity']
        else:
            q_bottom, vq_loss_bottom, info_bottom = self.quantizer_bottom(z_bottom)
            perplexity_bottom = info_bottom[0]
        
        # Decode
        recon = self.decoder(q_top, q_bottom, skip_128)
        
        # Compute losses
        losses = {
            'vq_top': vq_loss_top,
            'vq_bottom': vq_loss_bottom
        }
        
        # CMCR loss (if target provided and enabled)
        if self.use_cmcr and x_target is not None and self.training:
            cmcr_loss, cmcr_info = self.cmcr_module(
                z_source=(z_top, z_bottom),
                x_target=x_target,
                encodings_source=encodings_top
            )
            losses['cmcr'] = cmcr_loss * self.cmcr_weight
        
        # Orthogonal loss for codebook diversity
        ortho_loss = self.quantizer_top.compute_orthogonal_loss()
        losses['orthogonal'] = ortho_loss
        
        # Metrics for logging
        metrics = {
            'perplexity_top': perplexity_top,
            'perplexity_bottom': perplexity_bottom,
            'perplexity_avg': (perplexity_top + perplexity_bottom) / 2,
            'alive_top': alive_top,
            'quant_error_top': error_top.item() if isinstance(error_top, torch.Tensor) else error_top,
            'indices_top': indices_top
        }
        
        if self.use_fdhq:
            metrics['indices_low'] = info_bottom.get('indices_low')
            metrics['indices_high'] = info_bottom.get('indices_high')
        
        return recon, losses, metrics
    
    def get_total_vq_loss(self, losses):
        """Compute total VQ loss from loss dictionary."""
        total = losses['vq_top'] + losses['vq_bottom']
        if 'cmcr' in losses:
            total = total + losses['cmcr']
        if 'orthogonal' in losses:
            total = total + 10.0 * losses['orthogonal']
        return total
    
    def encode(self, x):
        """Encode image to quantized latents."""
        z_top, z_bottom, _ = self.encoder(x)
        q_top, _, _ = self.quantizer_top(z_top)
        
        if self.use_fdhq:
            q_bottom, _, _ = self.quantizer_bottom(z_bottom)
        else:
            q_bottom, _, _ = self.quantizer_bottom(z_bottom)
        
        return q_top, q_bottom
    
    def decode(self, q_top, q_bottom):
        """Decode from quantized latents."""
        return self.decoder(q_top, q_bottom)
