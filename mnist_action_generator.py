"""
MNIST Action Generator

A neural network that takes an MNIST digit image and a one-hot encoded action
in the range [-A, A] and outputs an MNIST digit image corresponding to 
(input_label + action) % 10 (if cyclic) or input_label + action (if not cyclic).

Architecture:
1. Encoder: CNN to extract features from input MNIST image
2. Action Fusion: Concatenate image features with action vector
3. Decoder: CNN to generate output MNIST image
4. Optional: Discriminator for adversarial training

Training:
- Primary: MSE loss for pixel-wise reconstruction
- Optional: Classification loss for digit class guidance
- Optional: Adversarial loss for better quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import os
from typing import Tuple, Optional, Dict, Any


class MNISTActionDataset(Dataset):
    """
    Dataset for MNIST action-based digit generation.
    
    For each sample:
    - Input: MNIST digit image + action
    - Target: MNIST digit image from class (input_label + action) % 10 (if cyclic)
    """
    
    def __init__(self, 
                 max_action: int = 2, 
                 transform: Optional[transforms.Compose] = None,
                 cyclic: bool = True,
                 train: bool = True,
                 data_dir: str = './data'):
        """
        Args:
            max_action: Maximum action value (actions in [-max_action, max_action])
            transform: Image transforms to apply
            cyclic: If True, use modular arithmetic (label + action) % 10
                   If False, discard samples that result in invalid labels
            train: Whether to use training or test split
            data_dir: Directory to store MNIST data
        """
        self.max_action = max_action
        self.action_space = 2 * max_action + 1  # [-max_action, ..., max_action]
        self.transform = transform
        self.cyclic = cyclic
        
        # Load MNIST dataset
        self.mnist = datasets.MNIST(
            root=data_dir, 
            train=train, 
            download=True, 
            transform=transforms.ToTensor()
        )
        
        # Build index for each digit class for fast sampling
        self.class_to_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.mnist):
            self.class_to_indices[label].append(idx)
        
        # Pre-filter valid samples if not cyclic
        if not cyclic:
            self.valid_indices = []
            for idx, (_, label) in enumerate(self.mnist):
                # Check if any action would result in valid label
                for action in range(-max_action, max_action + 1):
                    target_class = label + action
                    if 0 <= target_class <= 9:
                        self.valid_indices.append(idx)
                        break
        else:
            self.valid_indices = list(range(len(self.mnist)))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get original image and label
        original_idx = self.valid_indices[idx]
        img, label = self.mnist[original_idx]
        
        # Sample random action
        if self.cyclic:
            action = random.randint(-self.max_action, self.max_action)
            target_class = (label + action) % 10
        else:
            # Sample action that results in valid target class
            valid_actions = []
            for a in range(-self.max_action, self.max_action + 1):
                target_class = label + a
                if 0 <= target_class <= 9:
                    valid_actions.append(a)
            action = random.choice(valid_actions)
            target_class = label + action
        
        # One-hot encode action
        action_onehot = torch.zeros(self.action_space)
        action_onehot[action + self.max_action] = 1.0
        
        # Sample target image from target class
        target_indices = self.class_to_indices[target_class]
        target_idx = random.choice(target_indices)
        target_img, _ = self.mnist[target_idx]
        
        # Apply transforms if provided
        if self.transform:
            img = self.transform(img)
            target_img = self.transform(target_img)
        
        return {
            'input_img': img,
            'action': action_onehot,
            'target_img': target_img,
            'input_label': label,
            'action_value': action,
            'target_label': target_class
        }


class MNISTActionGenerator(nn.Module):
    """
    Generator network for MNIST action-based digit generation.
    
    Architecture:
    1. Encoder: CNN to extract features from input image
    2. Action Fusion: Combine image features with action vector
    3. Decoder: CNN to generate output image
    """
    
    def __init__(self, 
                 max_action: int = 2,
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 n_fusion_layers: int = 2):
        super(MNISTActionGenerator, self).__init__()
        
        self.max_action = max_action
        self.action_space = 2 * max_action + 1
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Image Encoder: Extract features from 28x28 MNIST image
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 7x7 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Flatten and project to latent space
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Action Fusion: Combine image features with action
        fusion_layers = []
        input_dim = latent_dim + self.action_space
        
        for i in range(n_fusion_layers):
            fusion_layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
        
        self.fusion = nn.Sequential(*fusion_layers)
        
        # Image Decoder: Generate output image
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 4, 4)),
            
            # 4x4 -> 7x7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, input_img: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_img: Input MNIST image [batch_size, 1, 28, 28]
            action: One-hot action vector [batch_size, action_space]
            
        Returns:
            Generated MNIST image [batch_size, 1, 28, 28]
        """
        # Encode input image
        img_features = self.encoder(input_img)
        
        # Fuse with action
        fused = torch.cat([img_features, action], dim=1)
        fused = self.fusion(fused)
        
        # Decode to output image
        output_img = self.decoder(fused)
        
        return output_img


class MNISTDiscriminator(nn.Module):
    """
    Discriminator network for adversarial training.
    """
    
    def __init__(self, max_action: int = 2):
        super(MNISTDiscriminator, self).__init__()
        
        self.max_action = max_action
        self.action_space = 2 * max_action + 1
        
        self.conv_layers = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 7x7 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 2x2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Action conditioning
        self.action_embedding = nn.Linear(self.action_space, 256)
        
        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2 + 256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            img: Image [batch_size, 1, 28, 28]
            action: Action vector [batch_size, action_space]
            
        Returns:
            Real/fake probability [batch_size, 1]
        """
        # Extract image features
        img_features = self.conv_layers(img)
        img_features = img_features.view(img_features.size(0), -1)
        
        # Embed action
        action_features = self.action_embedding(action)
        
        # Combine and classify
        combined = torch.cat([img_features, action_features], dim=1)
        output = self.classifier(combined)
        
        return output


class MNISTClassifier(nn.Module):
    """
    Simple CNN classifier for MNIST digits.
    """
    
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_classifier(train_loader: DataLoader, 
                    val_loader: DataLoader, 
                    device: torch.device,
                    epochs: int = 10) -> MNISTClassifier:
    """
    Train a simple MNIST classifier for evaluation.
    """
    classifier = MNISTClassifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        classifier.train()
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return classifier


def train_generator(model: MNISTActionGenerator,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   device: torch.device,
                   epochs: int = 50,
                   lr: float = 1e-3,
                   save_dir: str = './checkpoints',
                   use_adversarial: bool = False,
                   lambda_clf: float = 0.1,
                   lambda_adv: float = 0.01) -> Tuple[list, list]:
    """
    Train the MNIST action generator.
    
    Args:
        model: Generator model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints
        use_adversarial: Whether to use adversarial training
        lambda_clf: Weight for classification loss
        lambda_adv: Weight for adversarial loss
        
    Returns:
        Tuple of (train_losses, val_losses)
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    
    # Optional components
    discriminator = None
    classifier = None
    
    if use_adversarial:
        discriminator = MNISTDiscriminator(max_action=model.max_action).to(device)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)
    
    if lambda_clf > 0:
        # Train a simple classifier for evaluation
        print("Training classifier for digit class guidance...")
        classifier = train_classifier(train_loader, val_loader, device)
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training phase
        model.train()
        if discriminator:
            discriminator.train()
        
        train_loss = 0.0
        train_batches = 0
        
        for batch in train_loader:
            input_img = batch['input_img'].to(device)
            action = batch['action'].to(device)
            target_img = batch['target_img'].to(device)
            target_label = batch['target_label'].to(device)
            
            # Generate output
            output_img = model(input_img, action)
            
            # Reconstruction loss
            recon_loss = mse_criterion(output_img, target_img)
            total_loss = recon_loss
            
            # Classification loss
            if lambda_clf > 0 and classifier is not None:
                with torch.no_grad():
                    clf_loss = F.cross_entropy(classifier(output_img), target_label)
                total_loss = total_loss + lambda_clf * clf_loss
            
            # Adversarial loss
            if use_adversarial and discriminator is not None:
                # Train generator
                optimizer.zero_grad()
                fake_prob = discriminator(output_img, action)
                adv_loss = F.binary_cross_entropy(fake_prob, torch.ones_like(fake_prob))
                total_loss = total_loss + lambda_adv * adv_loss
                total_loss.backward()
                optimizer.step()
                
                # Train discriminator
                d_optimizer.zero_grad()
                
                # Real images
                real_prob = discriminator(target_img, action)
                real_loss = F.binary_cross_entropy(real_prob, torch.ones_like(real_prob))
                
                # Fake images
                with torch.no_grad():
                    fake_img = model(input_img, action)
                fake_prob = discriminator(fake_img, action)
                fake_loss = F.binary_cross_entropy(fake_prob, torch.zeros_like(fake_prob))
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                d_optimizer.step()
                
            else:
                # Standard training without adversarial loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            train_loss += recon_loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_img = batch['input_img'].to(device)
                action = batch['action'].to(device)
                target_img = batch['target_img'].to(device)
                
                output_img = model(input_img, action)
                loss = mse_criterion(output_img, target_img)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, os.path.join(save_dir, 'best_model.pth'))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses


def evaluate_model(model: MNISTActionGenerator,
                  test_loader: DataLoader,
                  device: torch.device,
                  classifier: Optional[MNISTClassifier] = None) -> Dict[str, float]:
    """
    Evaluate the model performance.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_samples = 0
    mse_loss = 0.0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_img = batch['input_img'].to(device)
            action = batch['action'].to(device)
            target_img = batch['target_img'].to(device)
            target_label = batch['target_label'].to(device)
            
            # Generate output
            output_img = model(input_img, action)
            
            # MSE loss
            mse_loss += F.mse_loss(output_img, target_img).item()
            
            # Classification accuracy
            if classifier is not None:
                predictions = classifier(output_img)
                predicted_labels = torch.argmax(predictions, dim=1)
                correct_predictions += (predicted_labels == target_label).sum().item()
            
            total_samples += input_img.size(0)
    
    avg_mse = mse_loss / len(test_loader)
    accuracy = correct_predictions / total_samples if classifier is not None else 0.0
    
    return {
        'mse_loss': avg_mse,
        'classification_accuracy': accuracy,
        'total_samples': total_samples
    }


def visualize_results(model: MNISTActionGenerator,
                     test_loader: DataLoader,
                     device: torch.device,
                     max_action: int = 2,
                     n_samples: int = 8) -> None:
    """
    Visualize model results.
    """
    model.eval()
    
    # Get a batch of test data
    batch = next(iter(test_loader))
    input_img = batch['input_img'][:n_samples].to(device)
    action = batch['action'][:n_samples].to(device)
    target_img = batch['target_img'][:n_samples].to(device)
    input_label = batch['input_label'][:n_samples]
    action_value = batch['action_value'][:n_samples]
    target_label = batch['target_label'][:n_samples]
    
    with torch.no_grad():
        output_img = model(input_img, action)
    
    # Convert to numpy for plotting
    input_img = input_img.cpu().numpy()
    output_img = output_img.cpu().numpy()
    target_img = target_img.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
    
    for i in range(n_samples):
        # Input image
        axes[0, i].imshow(input_img[i, 0], cmap='gray')
        axes[0, i].set_title(f'Input: {input_label[i].item()}')
        axes[0, i].axis('off')
        
        # Generated image
        axes[1, i].imshow(output_img[i, 0], cmap='gray')
        axes[1, i].set_title(f'Generated: {input_label[i].item()}+{action_value[i].item()}={target_label[i].item()}')
        axes[1, i].axis('off')
        
        # Target image
        axes[2, i].imshow(target_img[i, 0], cmap='gray')
        axes[2, i].set_title(f'Target: {target_label[i].item()}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main training script.
    """
    # Configuration
    config = {
        'max_action': 2,
        'cyclic': True,
        'batch_size': 64,
        'epochs': 50,
        'lr': 1e-3,
        'use_adversarial': False,
        'lambda_clf': 0.1,
        'lambda_adv': 0.01,
        'save_dir': './checkpoints_mnist_action'
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Create datasets
    train_dataset = MNISTActionDataset(
        max_action=config['max_action'],
        transform=transform,
        cyclic=config['cyclic'],
        train=True
    )
    
    val_dataset = MNISTActionDataset(
        max_action=config['max_action'],
        transform=transform,
        cyclic=config['cyclic'],
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = MNISTActionGenerator(
        max_action=config['max_action'],
        latent_dim=128,
        hidden_dim=256,
        n_fusion_layers=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_generator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config['epochs'],
        lr=config['lr'],
        save_dir=config['save_dir'],
        use_adversarial=config['use_adversarial'],
        lambda_clf=config['lambda_clf'],
        lambda_adv=config['lambda_adv']
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate model
    print("Evaluating model...")
    classifier = train_classifier(train_loader, val_loader, device)
    metrics = evaluate_model(model, val_loader, device, classifier)
    
    print(f"Evaluation Results:")
    print(f"MSE Loss: {metrics['mse_loss']:.4f}")
    print(f"Classification Accuracy: {metrics['classification_accuracy']:.4f}")
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(model, val_loader, device, config['max_action'])
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    model, train_losses, val_losses = main()
