"""
Loss functions for Enhanced CS-VQ-VAE.

Includes reconstruction losses, frequency-aware losses, and perceptual losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class FrequencyLoss(nn.Module):
    """
    Frequency-domain loss for medical images.
    
    Medical images have important high-frequency details (edges, boundaries)
    that MSE fails to capture adequately. This loss operates in the frequency
    domain to better preserve these details.
    """
    def __init__(self, low_freq_weight: float = 1.0, high_freq_weight: float = 2.0):
        super().__init__()
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight
        
    def forward(self, pred, target):
        """
        Compute frequency-domain loss.
        
        Uses 2D FFT to compare frequency components.
        """
        # 2D FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Phase spectrum (important for structure)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # Create frequency mask (low freq at center after shift)
        B, C, H, W = pred.shape
        Y, X = torch.meshgrid(
            torch.linspace(-1, 1, H, device=pred.device),
            torch.linspace(-1, 1, W, device=pred.device),
            indexing='ij'
        )
        freq_distance = torch.sqrt(X**2 + Y**2)
        
        # Low frequency mask (center)
        low_freq_mask = (freq_distance < 0.3).float()
        high_freq_mask = 1 - low_freq_mask
        
        # Magnitude loss (weighted by frequency)
        mag_loss_low = F.l1_loss(
            pred_mag * low_freq_mask, target_mag * low_freq_mask
        )
        mag_loss_high = F.l1_loss(
            pred_mag * high_freq_mask, target_mag * high_freq_mask
        )
        
        # Phase loss (critical for edges)
        phase_loss = F.l1_loss(pred_phase, target_phase)
        
        total_loss = (
            self.low_freq_weight * mag_loss_low +
            self.high_freq_weight * mag_loss_high +
            0.5 * phase_loss
        )
        
        return total_loss


class PerceptualLoss(nn.Module):
    """Wrapper around LPIPS perceptual loss."""
    def __init__(self, net: str = 'alex'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)
        # Freeze LPIPS network
        for param in self.lpips.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        """
        Compute LPIPS perceptual loss.
        
        Handles grayscale by expanding to 3 channels.
        """
        # LPIPS expects 3 channels
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        return self.lpips(pred, target).mean()


class GradientLoss(nn.Module):
    """
    Gradient/edge loss for sharp reconstructions.
    
    Encourages the model to preserve edges and boundaries.
    """
    def __init__(self):
        super().__init__()
        # Sobel filters
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).float().view(1, 1, 3, 3))
        
    def forward(self, pred, target):
        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        
        # Gradient magnitude loss
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


class GANLoss(nn.Module):
    """
    Adversarial loss for VQGAN.
    """
    def __init__(self, type='hinge'):
        super().__init__()
        self.type = type
        
    def forward(self, disc_pred, target_is_real, is_discriminator=True):
        """
        Calculate GAN loss.
        
        Args:
            disc_pred: Discriminator output logits
            target_is_real: Boolean, true if target is real image
            is_discriminator: Boolean, true if calculating loss for discriminator update
        """
        if self.type == 'hinge':
            if is_discriminator:
                if target_is_real:
                    return -torch.mean(torch.min(torch.zeros_like(disc_pred), -1.0 + disc_pred))
                else:
                    return -torch.mean(torch.min(torch.zeros_like(disc_pred), -1.0 - disc_pred))
            else:
                return -torch.mean(disc_pred)
        
        elif self.type == 'vanilla':
            target = torch.ones_like(disc_pred) if target_is_real else torch.zeros_like(disc_pred)
            return F.binary_cross_entropy_with_logits(disc_pred, target)
            
        else:
            raise ValueError(f"Unknown GAN loss type: {self.type}")


class CombinedLoss(nn.Module):
    """
    Combined reconstruction loss for Enhanced CS-VQ-VAE.
    
    Components:
    - Pixel loss (MSE or L1)
    - Perceptual loss (LPIPS)
    - Frequency loss (FFT-based)
    - Gradient loss (edge preservation)
    - Adversarial loss (VQGAN)
    """
    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        frequency_weight: float = 0.1,
        gradient_weight: float = 0.1,
        pixel_loss_type: str = 'l1',  # 'l1' or 'mse'
        gan_weight: float = 0.0,
        gan_loss_type: str = 'hinge'
    ):
        super().__init__()
        
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.frequency_weight = frequency_weight
        self.gradient_weight = gradient_weight
        self.gan_weight = gan_weight
        
        self.pixel_loss = nn.L1Loss() if pixel_loss_type == 'l1' else nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(net='alex')
        self.frequency_loss = FrequencyLoss()
        self.gradient_loss = GradientLoss()
        self.gan_loss = GANLoss(type=gan_loss_type)
        
    def forward(self, pred, target, disc_pred_fake=None, disc_pred_real=None, mode='generator', global_step=0, start_step=0):
        """
        Compute combined loss.
        
        Args:
            pred: Reconstructed image
            target: Real image
            disc_pred_fake: Discriminator prediction on fake image
            disc_pred_real: Discriminator prediction on real image (only for discriminator mode)
            mode: 'generator' or 'discriminator'
            global_step: Current training step
            start_step: Step to start adversarial training
        
        Returns:
            total_loss: Combined loss value
            components: Dictionary with individual loss terms
        """
        components = {}
        
        if mode == 'generator':
            # Pixel loss
            pixel = self.pixel_loss(pred, target)
            components['pixel'] = pixel.item()
            
            # Perceptual loss
            perceptual = self.perceptual_loss(pred, target)
            components['perceptual'] = perceptual.item()
            
            # Frequency loss
            frequency = self.frequency_loss(pred, target)
            components['frequency'] = frequency.item()
            
            # Gradient loss
            gradient = self.gradient_loss(pred, target)
            components['gradient'] = gradient.item()
            
            # Reconstruction loss (weighted sum)
            rec_loss = (
                self.pixel_weight * pixel +
                self.perceptual_weight * perceptual +
                self.frequency_weight * frequency +
                self.gradient_weight * gradient
            )
            
            # Adversarial loss
            adv_loss = torch.tensor(0.0, device=pred.device)
            if disc_pred_fake is not None and global_step >= start_step:
                adv_loss = self.gan_loss(disc_pred_fake, target_is_real=True, is_discriminator=False)
                components['adversarial'] = adv_loss.item()
                
            total = rec_loss + self.gan_weight * adv_loss
            return total, components
            
        elif mode == 'discriminator':
            if global_step < start_step:
                return torch.tensor(0.0, device=pred.device, requires_grad=True), {}
                
            # Real loss
            loss_real = self.gan_loss(disc_pred_real, target_is_real=True, is_discriminator=True)
            
            # Fake loss
            loss_fake = self.gan_loss(disc_pred_fake, target_is_real=False, is_discriminator=True)
            
            # Combined
            loss_d = 0.5 * (loss_real + loss_fake)
            components['disc_real'] = loss_real.item()
            components['disc_fake'] = loss_fake.item()
            components['disc_total'] = loss_d.item()
            
            return loss_d, components
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

