"""
Training script for Enhanced CS-VQ-VAE.

Features all novel contributions:
- EWSCS: Error-Weighted Semantic Coreset Selection
- CMCR: Cross-Modal Consistency Regularization
- FDHQ: Frequency-Decomposed Hierarchical Quantization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure

import numpy as np
from PIL import Image
import os
import yaml
import argparse
import logging
from math import log10
import matplotlib.pyplot as plt
from datetime import datetime

# Local imports
from models import EnhancedCSVQVAE, NLayerDiscriminator, weights_init
from losses import CombinedLoss


class MRIDataset(Dataset):
    """Dataset for paired MRI images (source and target modalities)."""
    
    def __init__(self, source_dir, target_dir=None, image_size=256):
        """
        Args:
            source_dir: Directory with source (post-contrast) images
            target_dir: Directory with target (pre-contrast) images (optional for CMCR)
        """
        self.source_paths = sorted([
            os.path.join(source_dir, f) for f in os.listdir(source_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self.target_paths = None
        if target_dir and os.path.exists(target_dir):
            self.target_paths = sorted([
                os.path.join(target_dir, f) for f in os.listdir(target_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            # Ensure same length
            min_len = min(len(self.source_paths), len(self.target_paths))
            self.source_paths = self.source_paths[:min_len]
            self.target_paths = self.target_paths[:min_len]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def __len__(self):
        return len(self.source_paths)
    
    def __getitem__(self, idx):
        source = Image.open(self.source_paths[idx])
        source = self.transform(source)
        
        if self.target_paths is not None:
            target = Image.open(self.target_paths[idx])
            target = self.transform(target)
            return source, target
        
        return source, source  # Return source twice if no target


def denormalize(x):
    """Denormalize tensor from [-1, 1] to [0, 1]."""
    return (x * 0.5 + 0.5).clamp(0, 1)


def calculate_psnr(original, reconstructed):
    """Calculate PSNR between two images."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100.0
    return 20 * log10(1.0 / np.sqrt(mse))


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_epoch(model, val_loader, criterion, device, logger, config=None):
    """Run validation epoch."""
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    n_batches = 0
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    with torch.no_grad():
        for source, target in val_loader:
            source = source.to(device)
            target = target.to(device)
            
            # Forward pass
            recon, losses, metrics = model(source, x_target=target)
            
            # Validating only reconstruction/VQ loss, not discriminator
            # For validation we simplify and assume generator mode without adversarial component for loss calculation
            # or we could calculate it if we carry discriminator around, but standard to just track Recon terms
            recon_loss, _ = criterion(recon, target) # This calculates pure reconstruction terms as adv loss requires disc inputs
            
            # total_loss in validation: Recon + VQ
            vq_loss = model.get_total_vq_loss(losses)
            total_loss = recon_loss + vq_loss
            
            running_loss += total_loss.item()
            
            # Metrics
            recon_denorm = denormalize(recon)
            target_denorm = denormalize(target)
            
            # PSNR
            for i in range(recon.size(0)):
                psnr = calculate_psnr(
                    target_denorm[i].cpu().numpy(),
                    recon_denorm[i].cpu().numpy()
                )
                running_psnr += psnr
            
            # SSIM
            ssim = ssim_metric(recon_denorm, target_denorm)
            running_ssim += ssim.item() * source.size(0)
            
            n_batches += 1
            
            if config and config['experiment'].get('dry_run', False) and n_batches >= 5:
                 logger.info("DRY RUN: Breaking validation loop early")
                 break
    
    n_samples = len(val_loader.dataset)
    if config and config['experiment'].get('dry_run', False):
        n_samples = n_batches * val_loader.batch_size # Approximate for dry run correctness
    
    return {
        'loss': running_loss / n_batches,
        'psnr': running_psnr / n_samples,
        'ssim': running_ssim / n_samples
    }


def train_epoch(model, discriminator, train_loader, optimizer_g, optimizer_d, criterion, config, device, epoch, logger):
    """Run one training epoch."""
    model.train()
    if discriminator:
        discriminator.train()
        
    running_loss_g = 0.0
    running_loss_d = 0.0
    running_perplexity = 0.0
    running_alive = 0.0
    n_batches = 0
    
    start_step = config['training']['discriminator']['start_epoch'] * len(train_loader)
    
    for batch_idx, (source, target) in enumerate(train_loader):
        global_step = (epoch - 1) * len(train_loader) + batch_idx
        
        source = source.to(device)
        target = target.to(device)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_g.zero_grad()
        
        # Forward pass
        recon, losses, metrics = model(source, x_target=target)
        
        # Calculate adversarial loss inputs if discriminator exists
        disc_pred_fake = None
        if discriminator:
            disc_pred_fake = discriminator(recon)
            
        # Reconstruction loss + Adversarial loss
        recon_loss, loss_components = criterion(
            recon, target, disc_pred_fake=disc_pred_fake, mode='generator', 
            global_step=global_step, start_step=start_step
        )
        
        # VQ losses
        vq_loss = model.get_total_vq_loss(losses)
        
        # Total generator loss
        total_loss_g = recon_loss + vq_loss
        
        # Backward G
        total_loss_g.backward()
        
        # Gradient clipping
        if config['training'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip']
            )
        
        optimizer_g.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        loss_d_item = 0.0
        if discriminator and global_step >= start_step:
            optimizer_d.zero_grad()
            
            # Detach recon to avoid backprop into generator
            disc_pred_fake_d = discriminator(recon.detach())
            disc_pred_real_d = discriminator(target)
            
            loss_d, d_components = criterion(
                None, None, disc_pred_fake=disc_pred_fake_d, disc_pred_real=disc_pred_real_d,
                mode='discriminator', global_step=global_step, start_step=start_step
            )
            
            loss_d.backward()
            optimizer_d.step()
            loss_d_item = loss_d.item()
        
        # Accumulate metrics
        running_loss_g += total_loss_g.item()
        running_loss_d += loss_d_item
        running_perplexity += metrics['perplexity_avg'].item()
        running_alive += metrics.get('alive_top', config['model']['num_embeddings'])
        n_batches += 1
        
        # Log
        if batch_idx % config['training']['log_interval'] == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss G: {total_loss_g.item():.4f} "
                f"Loss D: {loss_d_item:.4f} "
                f"Perp: {metrics['perplexity_avg'].item():.2f} "
                f"Alive: {metrics.get('alive_top', 0)}"
            )
        
        if config['experiment'].get('dry_run', False) and batch_idx >= 5:
            logger.info("DRY RUN: Breaking training loop early")
            break
    
    return {
        'loss_g': running_loss_g / n_batches,
        'loss_d': running_loss_d / n_batches,
        'perplexity': running_perplexity / n_batches,
        'alive': running_alive / n_batches
    }


def save_samples(model, val_loader, device, save_path, epoch):
    """Save sample reconstructions."""
    model.eval()
    
    with torch.no_grad():
        source, target = next(iter(val_loader))
        source = source[:4].to(device)
        target = target[:4].to(device)
        
        recon, _, _ = model(source)
        
        # Denormalize
        source = denormalize(source)
        target = denormalize(target)
        recon = denormalize(recon)
        
        # Create figure
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for i in range(4):
            axes[0, i].imshow(source[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title('Source (Post)')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(target[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title('Target (Pre)')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(recon[i, 0].cpu().numpy(), cmap='gray')
            axes[2, i].set_title('Reconstruction')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'samples_epoch_{epoch}.png'), dpi=150)
        plt.close()


def train(config_path, args):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    
    # Dry run overrides
    if args.dry_run:
        config['training']['num_epochs'] = 1
        config['training']['log_interval'] = 1
        config['training']['save_interval'] = 1
        config['training']['sample_interval'] = 1
        config['experiment']['name'] += "_dryrun"
        if 'experiment' not in config: config['experiment'] = {}
        config['experiment']['dry_run'] = True
    
    # Override config with command line args
    if args.exp:
        config['experiment']['name'] = args.exp
    if args.device is not None:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup directories
    exp_dir = os.path.join(
        config['experiment']['output_dir'],
        config['experiment']['name']
    )
    log_dir = os.path.join(exp_dir, 'logs')
    samples_dir = os.path.join(exp_dir, 'samples')
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")
    
    # Set seed
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    
    # Data paths
    source_folder = config['data'].get('source_folder', 'postcontrast')
    target_folder = config['data'].get('target_folder', 'precontrast')
    
    source_train_dir = os.path.join(config['data']['base_path'], 'train', source_folder)
    target_train_dir = os.path.join(config['data']['base_path'], 'train', target_folder)
    source_test_dir = os.path.join(config['data']['base_path'], 'test', source_folder)
    target_test_dir = os.path.join(config['data']['base_path'], 'test', target_folder)
    
    # Use validation set if it exists, otherwise split from train
    val_dir_exists = os.path.exists(os.path.join(config['data']['base_path'], 'val', source_folder))
    
    # Datasets
    train_dataset = MRIDataset(
        source_train_dir, 
        target_train_dir,
        image_size=config['data']['image_size']
    )
    
    if val_dir_exists:
        # Use dedicated val directory
        source_val_dir = os.path.join(config['data']['base_path'], 'val', source_folder)
        target_val_dir = os.path.join(config['data']['base_path'], 'val', target_folder)
        val_dataset = MRIDataset(
            source_val_dir,
            target_val_dir,
            image_size=config['data']['image_size']
        )
        train_size = len(train_dataset)
        val_size = len(val_dataset)
    else:
        # Split from train
        val_size = int(len(train_dataset) * config['data']['val_split'])
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    test_dataset = MRIDataset(
        source_test_dir,
        target_test_dir if config['model']['cmcr']['enabled'] else None,
        image_size=config['data']['image_size']
    )
    
    logger.info(f"Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Model
    model = EnhancedCSVQVAE(
        in_channels=config['model']['in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        latent_channels=config['model']['latent_channels'],
        num_embeddings=config['model']['num_embeddings'],
        commitment_cost=config['model']['commitment_cost'],
        error_weight=config['model']['ewscs']['error_weight'],
        num_semantic_clusters=config['model']['ewscs']['num_semantic_clusters'],
        use_fdhq=config['model']['fdhq']['enabled'],
        low_freq_weight=config['model']['fdhq']['low_freq_weight'],
        high_freq_weight=config['model']['fdhq']['high_freq_weight'],
        use_cmcr=config['model']['cmcr']['enabled'],
        cmcr_weight=config['model']['cmcr']['weight']
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Discriminator
    discriminator = None
    optimizer_d = None
    discriminator_config = config['training'].get('discriminator', {})
    if discriminator_config.get('enabled', False):
        discriminator = NLayerDiscriminator(input_nc=1).to(device)
        discriminator.apply(weights_init)
        logger.info(f"Discriminator initialized. Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
        
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=discriminator_config['learning_rate'],
            betas=(0.5, 0.999)
        )
    
    # Optimizer G
    optimizer_g = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=tuple(config['training']['betas']),
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler_type = config['training']['scheduler']['type']
    if scheduler_type == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer_g,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['scheduler']['min_lr']
        )
    elif scheduler_type == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer_g,
            mode='min',
            factor=0.5,
            patience=10
        )
    else:
        scheduler = None
    
    # Loss
    criterion = CombinedLoss(
        pixel_weight=config['loss']['pixel_weight'],
        perceptual_weight=config['loss']['perceptual_weight'],
        frequency_weight=config['loss']['frequency_weight'],
        gradient_weight=config['loss']['gradient_weight'],
        pixel_loss_type=config['loss']['pixel_loss_type'],
        gan_weight=discriminator_config['weight'] if discriminator_config.get('enabled') else 0.0,
        gan_loss_type=discriminator_config.get('gan_loss_type', 'hinge')
    ).to(device)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []
    val_history = []
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model, discriminator, train_loader, optimizer_g, optimizer_d, criterion, config, device, epoch, logger
        )
        train_history.append(train_metrics)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, logger, config)
        val_history.append(val_metrics)
        
        # Log
        logger.info(
            f"Epoch {epoch} - "
            f"Train Loss G: {train_metrics['loss_g']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val PSNR: {val_metrics['psnr']:.2f}, "
            f"Val SSIM: {val_metrics['ssim']:.4f}, "
            f"Perplexity: {train_metrics['perplexity']:.2f}"
        )
        
        # Scheduler step
        if scheduler_type == 'plateau':
            scheduler.step(val_metrics['loss'])
        elif scheduler is not None:
            scheduler.step()
        
        # Save samples
        if epoch % config['training']['sample_interval'] == 0:
            save_samples(model, val_loader, device, samples_dir, epoch)
        
        # Checkpointing
        if epoch % config['training']['save_interval'] == 0:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_psnr': val_metrics['psnr']
            }
            if discriminator:
                save_dict['discriminator_state_dict'] = discriminator.state_dict()
                save_dict['optimizer_d_state_dict'] = optimizer_d.state_dict()
                
            torch.save(save_dict, os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            logger.info(f"New best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            if patience_counter >= config['training']['early_stopping']['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Final evaluation
    logger.info("Running final evaluation on test set...")
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pth')))
    test_metrics = validate_epoch(model, test_loader, criterion, device, logger, config)
    
    logger.info(
        f"Final Test Results - "
        f"Loss: {test_metrics['loss']:.4f}, "
        f"PSNR: {test_metrics['psnr']:.2f} dB, "
        f"SSIM: {test_metrics['ssim']:.4f}"
    )
    
    # Save training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot([m['loss_g'] for m in train_history], label='Train G')
    plt.plot([m['loss'] for m in val_history], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot([m['psnr'] for m in val_history])
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR')
    
    plt.subplot(1, 3, 3)
    plt.plot([m['perplexity'] for m in train_history])
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Codebook Perplexity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    logger.info(f"Training complete! Results saved to {exp_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Enhanced CS-VQ-VAE')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment name (overrides config)')
    parser.add_argument('--device', type=int, default=None,
                        help='CUDA device ID')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run only 1 epoch for testing')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE: Setting epochs to 1, intervals to 1")
        # Load config to override
        config = load_config(args.config)
        config['training']['num_epochs'] = 1
        config['training']['log_interval'] = 1
        config['training']['save_interval'] = 1
        config['training']['sample_interval'] = 1
        # We need to pass the modified config to train/main logic
        # But train() currently loads config from path. 
        # Refactoring train() to accept config dict or we handle it inside train()
        # Let's handle it by patching train() to accept override_config
        
    train(args.config, args)
