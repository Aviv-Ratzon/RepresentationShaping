# CIFAR-100 Conditional Image Generation

This project implements a PyTorch-based conditional image generation model for CIFAR-100 that learns to transform images based on actions applied to latent variables.

## Overview

The model learns to:
- Encode CIFAR-100 images into a low-dimensional latent space
- Apply actions (discrete transformations) to the latent representations
- Decode the transformed latent vectors back to images
- Generate images conditioned on class indices and actions

## Key Features

- **Action Conditioning**: Each CIFAR-100 class has a latent variable equal to its class index
- **Discrete Actions**: Actions are one-hot encoded in the range [-A, A]
- **VAE Architecture**: Uses a Variational Autoencoder with action conditioning
- **Comprehensive Analysis**: Includes distance matrices, PCA plots, and sample visualizations
- **Automatic Checkpointing**: Saves model weights and generates visualizations at regular intervals

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. The script will automatically download CIFAR-100 dataset on first run.

## Usage

### Basic Training

```bash
python cifar100_conditional_generator.py --epochs 100 --batch_size 64 --latent_dim 32 --action_range 5
```

### Command Line Arguments

- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 64)
- `--latent_dim`: Latent space dimension (default: 32)
- `--action_range`: Action range [-A, A] (default: 5)
- `--lr`: Learning rate (default: 1e-3)
- `--beta`: KL divergence weight (default: 1.0)
- `--checkpoint_freq`: Checkpoint frequency in epochs (default: 10)
- `--output_dir`: Output directory for results (default: ./outputs)
- `--device`: Device to use (auto/cpu/cuda, default: auto)
- `--num_workers`: Number of data loader workers (default: 4)

### Testing

Run the test script to verify everything works:

```bash
python test_cifar100_generator.py
```

## Model Architecture

The model consists of:

1. **Encoder**: Convolutional network that maps images to latent space
2. **Action Embedding**: Linear layer that embeds one-hot actions
3. **Latent Manipulation**: Adds action embedding to latent representation
4. **Decoder**: Transposed convolutional network that reconstructs images

## Output Files

The training process generates:

- **Model Checkpoints**: `checkpoint_epoch_X.pth` - Model weights and optimizer state
- **Sample Grids**: `sample_grid_epoch_X.png` - Input, target, and generated images
- **Distance Matrices**: 
  - `distance_matrix_all_epoch_X.npy` - All sample distances
  - `distance_matrix_means_epoch_X.npy` - Class mean distances
  - `distance_matrices_epoch_X.png` - Visualized distance matrices
- **PCA Plots**: `pca_plots_epoch_X.png` - 2D PCA visualization of latent space
- **Training Curves**: `training_curves.png` - Loss curves over training

## Dataset Details

- **Input**: CIFAR-100 images (32x32x3)
- **Actions**: One-hot encoded discrete actions in range [-A, A]
- **Target**: CIFAR-100 images from class (original_class + action)
- **Filtering**: Samples where (class + action) is outside [0, 99] are discarded

## Example Usage

```bash
# Quick test run
python cifar100_conditional_generator.py --epochs 5 --batch_size 32 --latent_dim 16 --action_range 3

# Full training with custom parameters
python cifar100_conditional_generator.py \
    --epochs 200 \
    --batch_size 128 \
    --latent_dim 64 \
    --action_range 10 \
    --lr 5e-4 \
    --beta 0.5 \
    --checkpoint_freq 20 \
    --output_dir ./my_experiment
```

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- scikit-learn 1.0.0+
- matplotlib 3.4.0+
- seaborn 0.11.0+

## Notes

- The model uses a VAE architecture with reparameterization trick
- Actions are applied additively to the latent representation
- Class-specific latent dimensions are used for generation
- All visualizations are automatically generated during training
- The script handles CUDA/CPU device selection automatically
