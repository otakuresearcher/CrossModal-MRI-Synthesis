"""
Cross-Modal Consistency Regularization (CMCR) Module.

Novel contribution: Regularizes the latent space to encode modality-invariant
anatomical structure, not contrast-specific features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalConsistencyLoss(nn.Module):
    """
    Cross-Modal Consistency Regularization (CMCR).
    
    Novel Motivation:
    In cross-modal synthesis (post-contrast → pre-contrast), the latent
    representations should capture modality-invariant anatomical structure,
    not contrast-specific appearance. This loss encourages the encoder to
    produce similar latent distributions for corresponding regions in
    source and target modalities.
    
    Key Insight:
    A good codebook should encode ANATOMY, not CONTRAST. CMCR achieves this
    by minimizing the latent space distance between corresponding image pairs.
    
    Components:
    1. Feature-level alignment (L2 in latent space)
    2. Distribution alignment (KL divergence on quantized codes)
    3. Structural consistency (correlation-based)
    """
    def __init__(
        self,
        feature_weight: float = 1.0,
        distribution_weight: float = 0.5,
        structure_weight: float = 0.5,
        temperature: float = 0.1
    ):
        super().__init__()
        self.feature_weight = feature_weight
        self.distribution_weight = distribution_weight
        self.structure_weight = structure_weight
        self.temperature = temperature
        
    def feature_alignment_loss(self, z_source, z_target):
        """
        L2 alignment in latent feature space.
        
        Encourages the encoder to produce similar features for
        post-contrast and pre-contrast versions of the same anatomy.
        """
        # Normalize features for scale-invariant comparison
        z_source_norm = F.normalize(z_source.flatten(2), dim=-1)
        z_target_norm = F.normalize(z_target.flatten(2), dim=-1)
        
        return F.mse_loss(z_source_norm, z_target_norm)
    
    def distribution_alignment_loss(self, encodings_source, encodings_target):
        """
        KL divergence on the quantized code distributions.
        
        Ensures both modalities utilize the codebook similarly,
        indicating modality-invariant encoding.
        """
        # Compute code usage distributions
        p_source = encodings_source.mean(0) + 1e-8  # (K,)
        p_target = encodings_target.mean(0) + 1e-8  # (K,)
        
        # Symmetric KL divergence
        kl_forward = F.kl_div(
            torch.log(p_source), p_target, reduction='batchmean'
        )
        kl_backward = F.kl_div(
            torch.log(p_target), p_source, reduction='batchmean'
        )
        
        return (kl_forward + kl_backward) / 2
    
    def structural_consistency_loss(self, z_source, z_target):
        """
        Correlation-based structural consistency.
        
        Measures if the spatial structure in latent space is preserved
        across modalities (anatomy should have same relative arrangement).
        """
        B, C, H, W = z_source.shape
        
        # Flatten spatial dimensions
        z_s = z_source.view(B, C, -1)  # (B, C, H*W)
        z_t = z_target.view(B, C, -1)
        
        # Compute spatial correlation matrices
        z_s_centered = z_s - z_s.mean(dim=-1, keepdim=True)
        z_t_centered = z_t - z_t.mean(dim=-1, keepdim=True)
        
        # Cross-correlation (should be identity-like for consistent structure)
        # Using cosine similarity for stability
        z_s_norm = F.normalize(z_s_centered, dim=-1)
        z_t_norm = F.normalize(z_t_centered, dim=-1)
        
        # Channel-wise correlation
        correlation = torch.bmm(z_s_norm, z_t_norm.transpose(1, 2))  # (B, C, C)
        
        # Target is identity (same anatomy = same correlation pattern)
        identity = torch.eye(C, device=correlation.device).unsqueeze(0)
        
        return F.mse_loss(correlation, identity.expand(B, -1, -1))
    
    def forward(
        self, 
        z_source, 
        z_target, 
        encodings_source=None, 
        encodings_target=None
    ):
        """
        Compute CMCR loss.
        
        Args:
            z_source: Latent features from source modality (post-contrast)
            z_target: Latent features from target modality (pre-contrast)
            encodings_source: One-hot encodings from source (optional)
            encodings_target: One-hot encodings from target (optional)
            
        Returns:
            loss: Combined CMCR loss
            components: Dictionary with individual loss terms
        """
        loss = 0.0
        components = {}
        
        # Feature alignment
        feat_loss = self.feature_alignment_loss(z_source, z_target)
        loss = loss + self.feature_weight * feat_loss
        components['cmcr_feature'] = feat_loss.item()
        
        # Distribution alignment (if encodings provided)
        if encodings_source is not None and encodings_target is not None:
            dist_loss = self.distribution_alignment_loss(
                encodings_source, encodings_target
            )
            loss = loss + self.distribution_weight * dist_loss
            components['cmcr_distribution'] = dist_loss.item()
        
        # Structural consistency
        struct_loss = self.structural_consistency_loss(z_source, z_target)
        loss = loss + self.structure_weight * struct_loss
        components['cmcr_structure'] = struct_loss.item()
        
        return loss, components


class CMCREncoder(nn.Module):
    """
    Auxiliary encoder for CMCR that processes the target modality.
    
    This encoder is used during training to extract latent features
    from the target (pre-contrast) images for consistency regularization.
    It can be:
    1. A copy of the main encoder (shared weights) - simplest
    2. A frozen pre-trained encoder - for stronger regularization
    3. A separate trainable encoder - for adaptive consistency
    """
    def __init__(self, encoder, mode='shared'):
        """
        Args:
            encoder: The main encoder module
            mode: 'shared' (same weights), 'frozen' (detached), or 'separate'
        """
        super().__init__()
        self.mode = mode
        
        if mode == 'shared':
            self.encoder = encoder
        elif mode == 'frozen':
            self.encoder = encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:  # separate
            # Deep copy would go here
            self.encoder = encoder
            
    def forward(self, x):
        if self.mode == 'frozen':
            with torch.no_grad():
                return self.encoder(x)
        return self.encoder(x)


class CMCRModule(nn.Module):
    """
    Complete CMCR module for training.
    
    Combines the auxiliary encoder with the consistency loss.
    """
    def __init__(
        self,
        encoder,
        encoder_mode: str = 'shared',
        feature_weight: float = 1.0,
        distribution_weight: float = 0.5,
        structure_weight: float = 0.5
    ):
        super().__init__()
        
        self.aux_encoder = CMCREncoder(encoder, mode=encoder_mode)
        self.cmcr_loss = CrossModalConsistencyLoss(
            feature_weight=feature_weight,
            distribution_weight=distribution_weight,
            structure_weight=structure_weight
        )
        
    def forward(
        self,
        z_source,
        x_target,
        encodings_source=None
    ):
        """
        Compute CMCR loss for a batch.
        
        Args:
            z_source: Encoder output for source (post-contrast)
            x_target: Target images (pre-contrast)
            encodings_source: Codebook encodings for source
            
        Returns:
            loss: CMCR loss value
            components: Dictionary with loss components
        """
        # Encode target
        z_target_output = self.aux_encoder(x_target)
        
        # Handle hierarchical encoder output (may return 2 or 3 values)
        if isinstance(z_target_output, tuple):
            if len(z_target_output) == 3:
                z_target_top, z_target_bottom, _ = z_target_output  # skip is not used
            else:
                z_target_top, z_target_bottom = z_target_output
            
            # Use bottom (higher resolution) for CMCR
            z_target = z_target_bottom
            
            # If source is also tuple
            if isinstance(z_source, tuple):
                if len(z_source) == 3:
                    z_source = z_source[1]  # bottom
                else:
                    z_source = z_source[1]  # bottom
        else:
            z_target = z_target_output
            if isinstance(z_source, tuple):
                z_source = z_source[1] if len(z_source) >= 2 else z_source[0]
        
        # Compute CMCR loss
        loss, components = self.cmcr_loss(
            z_source, z_target.detach(),  # Detach target to not backprop through aux encoder
            encodings_source=encodings_source,
            encodings_target=None  # Would need to quantize target too
        )
        
        return loss, components
