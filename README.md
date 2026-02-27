# Enhanced CS-VQ-VAE

**Novel contributions for cross-modal MRI synthesis with improved codebook learning.**

## Novel Contributions

### 1. EWSCS - Error-Weighted Semantic Coreset Selection
Integrates reconstruction error as a priority signal in codebook re-initialization.

```
score(z_i) = λ * error(z_i) + (1-λ) * min_dist(z_i, E_live)
```

### 2. CMCR - Cross-Modal Consistency Regularization
Regularizes latent space to encode modality-invariant anatomical structure.

### 3. FDHQ - Frequency-Decomposed Hierarchical Quantization
Uses frequency-domain hierarchy with separate codebooks for low/high frequencies.

## Project Structure

```
enhanced-cs-vqvae/
├── models/
│   ├── ewscs_quantizer.py     # Error-weighted semantic coreset selection
│   ├── fdhq_quantizer.py      # Frequency-decomposed quantization
│   ├── cmcr_module.py         # Cross-modal consistency
│   └── enhanced_vqvae2.py     # Combined model
├── losses/
│   └── losses.py              # Frequency, perceptual, gradient losses
├── train.py                   # Training script
├── config.yaml                # Configuration
└── README.md                  # This file
```

## Quick Start

```bash
# Train with default config
python train.py --config config.yaml --exp "experiment_v1" --device 0

# Run ablation (disable FDHQ)
# Edit config.yaml: model.fdhq.enabled: false
python train.py --config config.yaml --exp "ablation_no_fdhq"
```

## Configuration

Edit `config.yaml` to:
- Adjust model hyperparameters
- Enable/disable specific contributions (EWSCS, CMCR, FDHQ)
- Tune loss weights
- Set data paths

## Expected Improvements

| Contribution | Expected Δ PSNR |
|-------------|-----------------|
| EWSCS | +0.5-1.0 dB |
| CMCR | +0.3-0.7 dB |
| FDHQ | +0.5-1.0 dB |
| **Combined** | **+1.5-2.5 dB** |

## Requirements

- PyTorch >= 1.10
- torchvision
- torchmetrics
- lpips
- einops
- scikit-learn
- PyYAML
- matplotlib
- PIL
