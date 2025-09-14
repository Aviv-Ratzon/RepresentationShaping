#!/usr/bin/env python3
"""
CIFAR-100 Conditional Image Generation with Action Conditioning

This script trains a model to generate CIFAR-100 images conditioned on actions
applied to latent variables. Each class has a latent variable equal to its class index,
and actions are applied to transform the latent space.

Usage:
    python cifar100_conditional_generator.py --epochs 100 --batch_size 64 --latent_dim 32 --action_range 5
"""

import argparse
import os
import pickle
import random
from pathlib import Path
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns


class CIFAR100ActionDataset(Dataset):
    """Custom dataset for CIFAR-100 with action conditioning."""
    
    def __init__(self, root: str, train: bool = True, action_range: int = 5, 
                 transform=None, num_classes: int = 100):
        self.dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
        self.action_range = action_range
        self.num_classes = num_classes
        self.valid_samples = self._filter_valid_samples()
        
    def _filter_valid_samples(self) -> List[Tuple[int, int, int]]:
        """Filter samples where (class + action) is within valid range."""
        valid_samples = []
        for idx in range(len(self.dataset)):
            class_idx = self.dataset[idx][1]
            for action in range(-self.action_range, self.action_range + 1):
                new_class = class_idx + action
                if 0 <= new_class < self.num_classes:
                    valid_samples.append((idx, class_idx, action))
        return valid_samples
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        img_idx, original_class, action = self.valid_samples[idx]
        image, _ = self.dataset[img_idx]
        
        # Get target image from the new class
        target_class = original_class + action
        target_idx = self._find_target_image(target_class)
        target_image, _ = self.dataset[target_idx]
        
        # Create one-hot encoded action
        action_onehot = torch.zeros(2 * self.action_range + 1)
        action_onehot[action + self.action_range] = 1.0
        
        return {
            'input_image': image,
            'target_image': target_image,
            'action': action_onehot,
            'original_class': original_class,
            'target_class': target_class,
            'action_value': action
        }
    
    def _find_target_image(self, target_class: int) -> int:
        """Find a random image from the target class."""
        class_indices = [i for i, (_, label) in enumerate(self.dataset) if label == target_class]
        return random.choice(class_indices)


class Encoder(nn.Module):
    """Encoder network that maps images to latent space."""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 32, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        
        self.latent_dim = latent_dim
        
        # Build encoder
        modules = []
        in_channels = input_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate the size after convolutions
        self.fc_input_size = self._get_conv_output_size()
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.fc_input_size, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_input_size, latent_dim)
    
    def _get_conv_output_size(self):
        """Calculate the output size after convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            dummy_output = self.encoder(dummy_input)
            return int(np.prod(dummy_output.size()[1:]))
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    """Decoder network that maps latent space back to images."""
    
    def __init__(self, latent_dim: int = 32, output_channels: int = 3, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.latent_dim = latent_dim
        
        # Calculate the size needed for the first conv layer
        self.fc_output_size = hidden_dims[0] * 4 * 4  # 4x4 spatial size
        
        # Project from latent to conv input
        self.fc = nn.Linear(latent_dim, self.fc_output_size)
        
        # Build decoder
        modules = []
        in_channels = hidden_dims[0]
        for i, h_dim in enumerate(hidden_dims[1:]):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        # Final layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Tanh()
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), -1, 4, 4)
        return self.decoder(h)


class ActionEmbedding(nn.Module):
    """Action embedding layer."""
    
    def __init__(self, action_dim: int, latent_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.embedding = nn.Linear(action_dim, latent_dim)
    
    def forward(self, action):
        return self.embedding(action)


class ConditionalVAE(nn.Module):
    """Conditional VAE with action conditioning."""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 32, 
                 action_range: int = 5, hidden_dims: List[int] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_range = action_range
        
        self.encoder = Encoder(input_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, input_channels, hidden_dims)
        self.action_embedding = ActionEmbedding(2 * action_range + 1, latent_dim)
    
    def forward(self, x, action):
        # Encode input image
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        
        # Add action embedding
        action_embed = self.action_embedding(action)
        z_conditioned = z + action_embed
        
        # Decode conditioned latent
        recon_x = self.decoder(z_conditioned)
        
        return recon_x, mu, logvar, z, z_conditioned
    
    def generate(self, class_indices, actions, device):
        """Generate images for given class indices and actions."""
        with torch.no_grad():
            # Create latent vectors based on class indices
            z = torch.zeros(len(class_indices), self.latent_dim, device=device)
            for i, class_idx in enumerate(class_indices):
                z[i, :min(self.latent_dim, 100)] = 0  # Initialize with zeros
                if class_idx < self.latent_dim:
                    z[i, class_idx] = 1.0  # Set class-specific dimension
            
            # Add action embedding
            action_embed = self.action_embedding(actions)
            z_conditioned = z + action_embed
            
            # Decode
            generated = self.decoder(z_conditioned)
            return generated


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function."""
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def get_data_loaders(batch_size: int, action_range: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Get training and validation data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CIFAR100ActionDataset(
        root='./data', train=True, action_range=action_range, transform=transform
    )
    val_dataset = CIFAR100ActionDataset(
        root='./data', train=False, action_range=action_range, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def generate_sample_grid(model, data_loader, device, epoch, output_dir, num_samples=16):
    """Generate and save sample grid."""
    model.eval()
    
    # Get a batch of samples
    batch = next(iter(data_loader))
    input_images = batch['input_image'][:num_samples].to(device)
    target_images = batch['target_image'][:num_samples].to(device)
    actions = batch['action'][:num_samples].to(device)
    
    with torch.no_grad():
        generated_images, _, _, _, _ = model(input_images, actions)
    
    # Create grid
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
    
    for i in range(num_samples):
        # Input image
        axes[0, i].imshow(input_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        axes[0, i].set_title(f'Input\nClass: {batch["original_class"][i].item()}')
        axes[0, i].axis('off')
        
        # Target image
        axes[1, i].imshow(target_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        axes[1, i].set_title(f'Target\nClass: {batch["target_class"][i].item()}')
        axes[1, i].axis('off')
        
        # Generated image
        axes[2, i].imshow(generated_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        axes[2, i].set_title(f'Generated\nAction: {batch["action_value"][i].item()}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_grid_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def compute_latent_analysis(model, data_loader, device, epoch, output_dir):
    """Compute and save latent space analysis."""
    model.eval()
    
    all_latents = []
    all_classes = []
    class_means = {}
    
    with torch.no_grad():
        for batch in data_loader:
            input_images = batch['input_image'].to(device)
            actions = batch['action'].to(device)
            classes = batch['original_class'].numpy()
            
            # Get latent representations
            _, _, _, z, _ = model(input_images, actions)
            
            all_latents.append(z.cpu().numpy())
            all_classes.extend(classes)
    
    all_latents = np.vstack(all_latents)
    all_classes = np.array(all_classes)
    
    # Compute class means
    for class_idx in range(100):
        class_mask = all_classes == class_idx
        if np.any(class_mask):
            class_means[class_idx] = np.mean(all_latents[class_mask], axis=0)
    
    # Distance matrix of all latent vectors (sorted by class)
    sorted_indices = np.argsort(all_classes)
    sorted_latents = all_latents[sorted_indices]
    distance_matrix_all = euclidean_distances(sorted_latents)
    
    # Distance matrix of class-mean latent vectors
    class_mean_latents = np.array([class_means[i] for i in range(100) if i in class_means])
    distance_matrix_means = euclidean_distances(class_mean_latents)
    
    # Save distance matrices
    np.save(os.path.join(output_dir, f'distance_matrix_all_epoch_{epoch}.npy'), distance_matrix_all)
    np.save(os.path.join(output_dir, f'distance_matrix_means_epoch_{epoch}.npy'), distance_matrix_means)
    
    # Plot distance matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.heatmap(distance_matrix_all, ax=ax1, cmap='viridis', cbar=True)
    ax1.set_title(f'Distance Matrix - All Samples (Epoch {epoch})')
    ax1.set_xlabel('Sample Index (sorted by class)')
    ax1.set_ylabel('Sample Index (sorted by class)')
    
    sns.heatmap(distance_matrix_means, ax=ax2, cmap='viridis', cbar=True)
    ax2.set_title(f'Distance Matrix - Class Means (Epoch {epoch})')
    ax2.set_xlabel('Class Index')
    ax2.set_ylabel('Class Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distance_matrices_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # PCA plots
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_latents)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA of all samples
    scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=all_classes, cmap='tab20', alpha=0.6)
    ax1.set_title(f'PCA - All Samples (Epoch {epoch})')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=ax1)
    
    # PCA of class means
    if len(class_means) > 0:
        class_mean_array = np.array(list(class_means.values()))
        class_indices = np.array(list(class_means.keys()))
        pca_means = pca.transform(class_mean_array)
        
        scatter2 = ax2.scatter(pca_means[:, 0], pca_means[:, 1], c=class_indices, cmap='tab20', s=100)
        ax2.set_title(f'PCA - Class Means (Epoch {epoch})')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pca_plots_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Latent analysis saved for epoch {epoch}")


def train_epoch(model, train_loader, optimizer, device, beta=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_images = batch['input_image'].to(device)
        target_images = batch['target_image'].to(device)
        actions = batch['action'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_images, mu, logvar, z, z_conditioned = model(input_images, actions)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(recon_images, target_images, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                  f'Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')
    
    return total_loss / len(train_loader), total_recon_loss / len(train_loader), total_kl_loss / len(train_loader)


def validate(model, val_loader, device, beta=1.0):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_images = batch['input_image'].to(device)
            target_images = batch['target_image'].to(device)
            actions = batch['action'].to(device)
            
            recon_images, mu, logvar, z, z_conditioned = model(input_images, actions)
            loss, recon_loss, kl_loss = vae_loss(recon_images, target_images, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    return total_loss / len(val_loader), total_recon_loss / len(val_loader), total_kl_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 Conditional Image Generation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--action_range', type=int, default=5, help='Action range [-A, A]')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0, help='KL divergence weight')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Checkpoint frequency (epochs)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get data loaders
    print("Loading CIFAR-100 dataset...")
    train_loader, val_loader = get_data_loaders(args.batch_size, args.action_range, args.num_workers)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = ConditionalVAE(
        input_channels=3,
        latent_dim=args.latent_dim,
        action_range=args.action_range
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, args.beta)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_recon, val_kl = validate(model, val_loader, device, args.beta)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        
        # Save checkpoint and visualizations
        if (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint(model, optimizer, epoch + 1, val_loss, args.output_dir)
            generate_sample_grid(model, val_loader, device, epoch + 1, args.output_dir)
            compute_latent_analysis(model, val_loader, device, epoch + 1, args.output_dir)
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, val_loss, args.output_dir)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
