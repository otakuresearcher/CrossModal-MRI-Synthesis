import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure

import numpy as np
from PIL import Image
import os
import yaml
import argparse
import logging
from math import log10
from tqdm import tqdm

# Local imports
from models import EnhancedCSVQVAE

# Re-implementing necessary components from train.py that might not be easily importable or need modification
class TestMRIDataset(Dataset):
    """Dataset for paired MRI images (source and target modalities) for testing."""
    
    def __init__(self, source_dir, target_dir=None, image_size=256):
        """
        Args:
            source_dir: Directory with source (post-contrast) images
            target_dir: Directory with target (pre-contrast) images (optional)
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
        source_path = self.source_paths[idx]
        filename = os.path.basename(source_path)
        
        source = Image.open(source_path)
        source = self.transform(source)
        
        if self.target_paths is not None:
            target = Image.open(self.target_paths[idx])
            target = self.transform(target)
            return source, target, filename
        
        return source, source, filename # Return source twice if no target

def denormalize(x):
    """Denormalize tensor from [-1, 1] to [0, 1]."""
    return (x * 0.5 + 0.5).clamp(0, 1)

def calculate_psnr(original, reconstructed):
    """Calculate PSNR between two images."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100.0
    return 20 * log10(1.0 / np.sqrt(mse))

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(args):
    """Main evaluation function."""
    # Load config
    config = load_config(args.config)
    
    if args.device is not None:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    print(f"Using device: {device}")

    # Data paths
    source_folder = config['data'].get('source_folder', 'postcontrast')
    target_folder = config['data'].get('target_folder', 'precontrast')
    
    source_test_dir = os.path.join(config['data']['base_path'], 'test', source_folder)
    target_test_dir = os.path.join(config['data']['base_path'], 'test', target_folder)
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to a folder parallel to input/ground truth if possible, or inside experiment dir
        output_dir = os.path.join(config['data']['base_path'], 'test', 'generated_' + target_folder)
    
    # Create subdirectories
    input_dir = os.path.join(output_dir, 'input')
    generated_dir = os.path.join(output_dir, 'generated')
    ground_truth_dir = os.path.join(output_dir, 'ground_truth')
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)
    
    print(f"Saving images to: {output_dir}")

    # Dataset and Loader
    test_dataset = TestMRIDataset(
        source_test_dir,
        target_test_dir,
        image_size=config['data']['image_size']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, # Process one by one for saving
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"Test set size: {len(test_dataset)}")

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
    
    # Load Checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Try to find it in default experiment location
        exp_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['name'])
        checkpoint_path = os.path.join(exp_dir, 'best_model.pth')
        
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        # Handle both full checkpoint dict and just state_dict
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model.eval()
    
    # Metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    running_psnr = 0.0
    running_ssim = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for source, target, filename in tqdm(test_loader, desc="Evaluating"):
            source = source.to(device)
            target = target.to(device)
            
            # Forward pass
            recon, _, _ = model(source)
            
            # Denormalize
            source_denorm = denormalize(source)
            target_denorm = denormalize(target)
            recon_denorm = denormalize(recon)
            
            # Save Images
            fname = filename[0]
            
            # Helper to save tensor as image
            def save_tensor_img(tensor, path):
                img_np = tensor[0, 0].cpu().numpy() # (H, W)
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                img_pil.save(path)

            save_tensor_img(source_denorm, os.path.join(input_dir, fname))
            save_tensor_img(recon_denorm, os.path.join(generated_dir, fname))
            save_tensor_img(target_denorm, os.path.join(ground_truth_dir, fname))
            
            # Calculate Metrics
            psnr = calculate_psnr(
                target_denorm[0, 0].cpu().numpy(),
                recon_denorm[0, 0].cpu().numpy()
            )
            running_psnr += psnr
            
            ssim = ssim_metric(recon_denorm, target_denorm)
            running_ssim += ssim.item()
            
            n_samples += 1
            
    avg_psnr = running_psnr / n_samples
    avg_ssim = running_ssim / n_samples
    
    print(f"Evaluation Complete!")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    # Save metrics to a file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Enhanced CS-VQ-VAE')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save generated images')
    parser.add_argument('--device', type=int, default=None,
                        help='CUDA device ID')
    
    args = parser.parse_args()
    evaluate(args)
