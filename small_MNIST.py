#!/usr/bin/env python3
"""
MNIST Digit Generation with Action-based Transformation

This script implements a GAN-based model for generating MNIST digits by applying
actions to input digits. The model consists of:
- Encoder E(x, a): Maps input image + action to latent vector z
- Generator G(z): Maps latent vector to output image
- Discriminator D(x): Classifies images as digits 0-9 or fake (class 10)

Usage:
    python train.py --epochs 50 --N 1000 --A 2 --cyclic True
"""

import argparse
import os
import random
import shutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

# Set font family with fallback - suppress font warnings
import matplotlib
import matplotlib.font_manager as fm

# Suppress font warnings
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Configure fonts to avoid Times New Roman issues
matplotlib.rcParams['font.family'] = ['DejaVu Serif', 'serif']
matplotlib.rcParams['font.serif'] = ['DejaVu Serif', 'serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9

# Clear font cache to avoid conflicts
try:
    fm._rebuild()
except:
    pass

class MNISTActionDataset(Dataset):
    """
    Dataset for MNIST with action-based transformations.
    
    Args:
        N: Number of samples per digit class
        A: Action range [-A, A]
        cyclic: Whether to use cyclic addition (mod 10)
        transform: Image transformations
    """
    
    def __init__(self, N=10, A=2, cyclic=False, transform=None, image_actions=False):
        self.N = N
        self.A = A
        self.cyclic = cyclic
        self.transform = transform
        self.image_actions = image_actions
        
        # Load MNIST dataset
        self.dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        
        # Create balanced dataset with N samples per class
        self.data, self.labels = self._create_balanced_dataset()
        
        # If using image actions, create action image pool by digit class
        if self.image_actions:
            self.action_images_by_class = self._create_action_image_pool()
        
        # Generate action-target pairs
        self.samples = self._generate_samples()
        
        # Create index for fast lookup by target_label
        self.target_label_index = self._create_target_label_index()
        
        print(f"Created dataset with {len(self.samples)} samples")
        print(f"Action range: [-{A}, {A}], Cyclic: {cyclic}, Image actions: {image_actions}")
    
    def _create_balanced_dataset(self):
        """Create balanced dataset with N samples per digit class."""
        data_by_class = {i: [] for i in range(10)}
        
        # Group data by class
        for idx, (image, label) in enumerate(self.dataset):
            data_by_class[label].append((image, label))
        
        # Sample N examples per class
        balanced_data = []
        balanced_labels = []
        
        for digit in range(10):
            class_data = data_by_class[digit]
            if len(class_data) < self.N:
                print(f"Warning: Only {len(class_data)} samples available for digit {digit}, using all")
                selected = class_data
            else:
                selected = random.sample(class_data, self.N)
            
            for image, label in selected:
                balanced_data.append(image)
                balanced_labels.append(label)
        
        return balanced_data, balanced_labels
    
    def _create_action_image_pool(self):
        """Create a pool of action images organized by digit class."""
        action_images_by_class = {i: [] for i in range(10)}
        
        # Group all MNIST images by class
        for idx, (image, label) in enumerate(self.dataset):
            action_images_by_class[label].append(image)
        
        return action_images_by_class
    
    def _get_action_image(self, action_value):
        """Get a random action image corresponding to the absolute value of the action."""
        # Get the digit corresponding to the absolute value of the action
        action_digit = abs(action_value)
        if action_digit > 9:
            action_digit = action_digit % 10
        
        # Get sign: 1 for positive/zero, -1 for negative
        sign = 1 if action_value >= 0 else -1
        
        # Get a random image from the action digit class
        if len(self.action_images_by_class[action_digit]) > 0:
            action_image = random.choice(self.action_images_by_class[action_digit])
        else:
            # Fallback: create a zero image if no images available
            action_image = torch.zeros_like(self.data[0])
        
        return action_image, sign
    
    def _generate_samples(self):
        """Generate (input_image, input_label, action, target_label) samples."""
        samples = []
        
        for i, (image, input_label) in enumerate(zip(self.data, self.labels)):
            for action in np.arange(-self.A, self.A + 1):    
                # Compute target label
                if self.cyclic:
                    target_label = (input_label + action) % 10
                else:
                    target_label = input_label + action
                    # Skip if target is outside valid range
                    if target_label < 0 or target_label > 9:
                        continue
                
                # One-hot encode action (for non-image actions)
                action_onehot = torch.zeros(2 * self.A + 1)
                action_onehot[action + self.A] = 1.0
                
                sample_dict = {
                    'input_image': image,
                    'input_label': input_label,
                    'action': action,
                    'action_onehot': action_onehot,
                    'target_label': target_label
                }
                
                # If using image actions, add action image and sign
                if self.image_actions:
                    action_image, sign = self._get_action_image(action)
                    sample_dict['action_image'] = action_image
                    sample_dict['action_sign'] = sign
                
                samples.append(sample_dict)
        
        return samples
    
    def _create_target_label_index(self):
        """Create index mapping target_label -> list of sample indices for fast lookup."""
        index = {i: [] for i in range(10)}
        for idx, sample in enumerate(self.samples):
            target_label = sample['target_label']
            if isinstance(target_label, torch.Tensor):
                target_label = target_label.item()
            index[target_label].append(idx)
        return index
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.image_actions:
            return (
                sample['input_image'],
                sample['input_label'],
                sample['action_image'],
                sample['action_sign'],
                sample['target_label']
            )
        else:
            return (
                sample['input_image'],
                sample['input_label'],
                sample['action_onehot'],
                sample['target_label']
            )


class Encoder(nn.Module):
    """
    Encoder E(x, a) that maps input image and action to latent vector z.
    
    Args:
        input_channels: Number of input channels (1 for MNIST)
        action_dim: Dimension of action vector (2*A+1) - only used for non-image actions
        latent_dim: Dimension of latent vector z
        image_actions: Whether actions are images (True) or one-hot vectors (False)
    """
    
    def __init__(self, input_channels=1, action_dim=5, latent_dim=64, n_layers=2, dropout_rate=0.0, image_actions=False):
        super(Encoder, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.image_actions = image_actions
        
        # Observation image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),  # 28x28 -> 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 -> 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 7x7 -> 3x3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 3, 1, 0),  # 3x3 -> 1x1
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        if self.image_actions:
            # Action image encoder (same architecture as observation encoder)
            self.action_image_encoder = nn.Sequential(
                nn.Conv2d(input_channels, 32, 4, 2, 1),  # 28x28 -> 14x14
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 -> 7x7
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(64, 128, 4, 2, 1),  # 7x7 -> 3x3
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(128, 256, 3, 1, 0),  # 3x3 -> 1x1
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
            )
            
            # Sign processing layer (processes sign after action image encoding)
            self.sign_processor = nn.Sequential(
                nn.Linear(256 + 1, 64),  # 256 (action features) + 1 (sign)
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(64, 64),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            )
        else:
            # Action encoder for one-hot vectors
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(64, 64),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            )
        
        # Combined encoder
        combined_layers = []
        input_dim = 256 + 64  # image features + action features
        hidden_dim = 512
        for i in range(n_layers - 1):
            combined_layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            combined_layers.append(nn.LeakyReLU(0.2))
            if dropout_rate > 0:
                combined_layers.append(nn.Dropout(dropout_rate))
        combined_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.combined_encoder = nn.Sequential(*combined_layers)
    
    def forward(self, x, a, sign=None):
        # Encode observation image
        img_features = self.image_encoder(x)  # [B, 256, 1, 1]
        img_features = img_features.view(img_features.size(0), -1)  # [B, 256]
        
        # Encode action
        if self.image_actions:
            # Encode action image
            action_img_features = self.action_image_encoder(a)  # [B, 256, 1, 1]
            action_img_features = action_img_features.view(action_img_features.size(0), -1)  # [B, 256]
            
            # Process sign after action image encoding
            # sign should be [B, 1] with values -1 or 1
            if sign is None:
                raise ValueError("sign must be provided when image_actions=True")
            sign = sign.view(-1, 1).float()  # Ensure [B, 1] shape
            
            # Concatenate action image features with sign
            action_with_sign = torch.cat([action_img_features, sign], dim=1)  # [B, 257]
            
            # Process through sign processor
            action_features = self.sign_processor(action_with_sign)  # [B, 64]
        else:
            # Encode one-hot action vector
            action_features = self.action_encoder(a)  # [B, 64]
        
        # Combine features
        combined = torch.cat([img_features, action_features], dim=1)  # [B, 320]
        z = self.combined_encoder(combined)  # [B, latent_dim]
        
        return z


class Generator(nn.Module):
    """
    Generator G(z) that maps latent vector to output image.
    
    Args:
        latent_dim: Dimension of latent vector z
        output_channels: Number of output channels (1 for MNIST)
    """
    
    def __init__(self, latent_dim=64, output_channels=1):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 7 * 7 * 128),
            nn.LeakyReLU(0.2),
        )
        
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 14x14 -> 28x28
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, output_channels, 3, 1, 1),
            nn.Tanh(),
        )
    
    def forward(self, z):
        x = self.decoder(z)  # [B, 7*7*128]
        x = x.view(x.size(0), 128, 7, 7)  # [B, 128, 7, 7]
        x = self.conv_decoder(x)  # [B, 1, 28, 28]
        return x


class Discriminator(nn.Module):
    """
    Discriminator D(x) that classifies images into 11 classes (0-9 + fake).
    
    Args:
        input_channels: Number of input channels (1 for MNIST)
        num_classes: Number of classes (11: 0-9 + fake)
    """
    
    def __init__(self, input_channels=1, num_classes=11):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),  # 28x28 -> 14x14
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 -> 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 7x7 -> 3x3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 3, 1, 0),  # 3x3 -> 1x1
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        features = self.conv_layers(x)  # [B, 256, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 256]
        logits = self.classifier(features)  # [B, 11]
        return logits


class MNISTActionGAN:
    """
    Main GAN class for MNIST digit generation with actions.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        
        # Create output directories
        self.create_directories()
        
        # Initialize models
        self.setup_models()
        
        # Setup optimizers
        self.setup_optimizers()
        
        # Setup data
        self.setup_data()
        
        # Initialize loss tracking
        self.loss_history = {
            'd_loss': [],
            'g_loss': [],
            'real_loss': [],
            'fake_loss': [],
            'recon_loss': []
        }
        
        # Set regularization parameters based on type
        self._set_regularization_params()
        
        print(f"Using device: {self.device}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    def _set_regularization_params(self):
        """Set regularization parameters based on regularization type."""
        if self.args.regularization_type == 'l1':
            self.args.l1_reg = 0.001
            self.args.l2_reg = 0.0
            self.args.dropout_rate = 0.0
        elif self.args.regularization_type == 'l2':
            self.args.l1_reg = 0.0
            self.args.l2_reg = 0.001
            self.args.dropout_rate = 0.0
        elif self.args.regularization_type == 'dropout':
            self.args.l1_reg = 0.0
            self.args.l2_reg = 0.0
            self.args.dropout_rate = 0.3
        elif self.args.regularization_type == 'l1_l2':
            self.args.l1_reg = 0.0005
            self.args.l2_reg = 0.0005
            self.args.dropout_rate = 0.0
        elif self.args.regularization_type == 'all':
            self.args.l1_reg = 0.0003
            self.args.l2_reg = 0.0003
            self.args.dropout_rate = 0.2
        # For 'none', keep default values (all 0.0)
        
        print(f"Regularization settings: L1={self.args.l1_reg}, L2={self.args.l2_reg}, Dropout={self.args.dropout_rate}")
    
    def create_directories(self):
        """Create output directories."""
        base_dir = f"MNIST_small/{self.args.run_directory}"
        self.checkpoint_dir = f"{base_dir}/checkpoints"
        self.samples_dir = f"{base_dir}/samples"
        self.plots_dir = f"{base_dir}/plots"
        
        # Clear output folder if it exists
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
            print(f"Cleared existing output directory: {base_dir}")
        
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print(f"Results will be saved to: {base_dir}")
    
    def setup_models(self):
        """Initialize encoder, generator, and discriminator."""
        action_dim = 2 * self.args.A + 1
        
        self.encoder = Encoder(
            input_channels=1,
            action_dim=action_dim,
            latent_dim=self.args.latent_dim,
            n_layers=self.args.n_layers,
            dropout_rate=self.args.dropout_rate,
            image_actions=self.args.image_actions
        ).to(self.device)
        
        self.generator = Generator(
            latent_dim=self.args.latent_dim,
            output_channels=1
        ).to(self.device)
        
        self.discriminator = Discriminator(
            input_channels=1,
            num_classes=11
        ).to(self.device)
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            self.encoder = DataParallel(self.encoder)
            self.generator = DataParallel(self.generator)
            self.discriminator = DataParallel(self.discriminator)
    
    def setup_optimizers(self):
        """Setup optimizers for all models."""
        # Use different learning rates if specified
        g_lr = self.args.g_lr if self.args.g_lr is not None else self.args.lr
        d_lr = self.args.d_lr if self.args.d_lr is not None else self.args.lr
        
        self.optimizer_E = optim.Adam(self.encoder.parameters(), lr=g_lr, betas=(0.5, 0.999))
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
        
        # Setup learning rate schedulers
        if self.args.use_lr_scheduler:
            self.scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_G, mode='min', factor=0.5, patience=10, verbose=True
            )
            self.scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_D, mode='min', factor=0.5, patience=10, verbose=True
            )
        else:
            self.scheduler_G = None
            self.scheduler_D = None
    
    def setup_data(self):
        """Setup data loaders."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.dataset = MNISTActionDataset(
            N=self.args.N,
            A=self.args.A,
            cyclic=self.args.cyclic,
            transform=transform,
            image_actions=self.args.image_actions
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def train_discriminator(self, real_images, real_labels, fake_images, fake_labels):
        """Train discriminator for one step."""
        self.optimizer_D.zero_grad()
        
        batch_size = real_images.size(0)
        
        # Real images: classify as their true labels (0-9)
        real_logits = self.discriminator(real_images)
        real_loss = F.cross_entropy(real_logits, real_labels)
        
        # Fake images: classify as fake (class 10)
        fake_logits = self.discriminator(fake_images)
        fake_labels_tensor = torch.full((batch_size,), 10, dtype=torch.long, device=self.device)
        fake_loss = F.cross_entropy(fake_logits, fake_labels_tensor)
        
        d_loss = real_loss + fake_loss
        
        # Add gradient penalty if enabled
        if self.args.use_gradient_penalty:
            gp = self._compute_gradient_penalty(real_images, fake_images)
            d_loss += self.args.gp_weight * gp
        
        d_loss = self.args.d_loss_weight * d_loss
        d_loss.backward()
        
        # Apply gradient clipping
        if self.args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.args.gradient_clip)
        
        self.optimizer_D.step()
        
        return d_loss.item(), real_loss.item(), fake_loss.item()
    
    def _unpack_batch(self, batch):
        """Unpack batch, handling both image_actions and regular actions."""
        if self.args.image_actions:
            input_images, input_labels, action_images, action_signs, target_labels = batch
            return input_images, input_labels, action_images, action_signs, target_labels
        else:
            input_images, input_labels, actions, target_labels = batch
            return input_images, input_labels, actions, None, target_labels
    
    def _encode(self, input_images, actions, action_signs=None):
        """Encode input images and actions, handling both image_actions and regular actions."""
        if self.args.image_actions:
            return self.encoder(input_images, actions, action_signs)
        else:
            return self.encoder(input_images, actions)
    
    def train_generator_encoder(self, input_images, input_labels, actions, target_labels, action_signs=None):
        """Train generator and encoder for one step."""
        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()
        
        # Encode input image and action to latent
        z = self._encode(input_images, actions, action_signs)
        
        # Generate fake images
        fake_images = self.generator(z)
        
        # Standard GAN loss: fake images should be classified as real (not fake)
        fake_logits = self.discriminator(fake_images)
        fake_labels_tensor = torch.full((fake_logits.size(0),), 10, dtype=torch.long, device=self.device)
        
        # Generator wants to fool discriminator (make fake images look real)
        # So we want fake_logits to NOT be classified as class 10 (fake)
        gan_loss = F.cross_entropy(fake_logits, target_labels)  # Original loss for digit classification
        
        # Add adversarial loss: generator wants discriminator to think fake images are real
        # We want the "fake" probability (class 10) to be low
        fake_prob = F.softmax(fake_logits, dim=1)[:, 10]  # Probability of being fake
        adversarial_loss = fake_prob.mean()  # Minimize probability of being fake
        
        total_loss = gan_loss + self.args.adversarial_weight * adversarial_loss
        
        # Feature matching loss if enabled
        if self.args.use_feature_matching:
            # Get real images for feature matching
            real_images = self._get_reconstruction_targets(target_labels)
            real_features = self._extract_discriminator_features(real_images)
            fake_features = self._extract_discriminator_features(fake_images)
            fm_loss = F.mse_loss(fake_features, real_features)
            total_loss += self.args.feature_matching_weight * fm_loss
        else:
            fm_loss = torch.tensor(0.0)
        
        # Optional reconstruction loss
        if self.args.lambda_recon > 0:
            # Get random real images from target classes
            recon_images = self._get_reconstruction_targets(target_labels)
            recon_loss = F.mse_loss(fake_images, recon_images)
            total_loss += self.args.lambda_recon * recon_loss
        else:
            recon_loss = torch.tensor(0.0)
        
        # Add regularization loss
        reg_loss = self._compute_regularization_loss()
        total_loss += reg_loss
        
        total_loss = self.args.g_loss_weight * total_loss
        total_loss.backward()
        
        # Apply gradient clipping
        if self.args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.gradient_clip)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.gradient_clip)
        
        self.optimizer_G.step()
        self.optimizer_E.step()
        
        return total_loss.item(), gan_loss.item(), recon_loss.item()
    
    def _get_reconstruction_targets(self, target_labels):
        """Get random real images from target classes for reconstruction loss."""
        batch_size = target_labels.size(0)
        recon_images = torch.zeros_like(self.dataset.data[0]).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Use pre-built index for O(1) lookup instead of O(n) search
        target_label_index = self.dataset.target_label_index
        
        for i, target_label in enumerate(target_labels):
            target_label_val = target_label.item()
            
            # Fast lookup using index
            if target_label_val in target_label_index and target_label_index[target_label_val]:
                # Randomly select one from the pre-indexed list
                random_idx = random.choice(target_label_index[target_label_val])
                recon_images[i] = self.dataset.samples[random_idx]['input_image']
        
        return recon_images.to(self.device)
    
    def _compute_gradient_penalty(self, real_images, fake_images):
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        # Interpolate between real and fake images
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated images
        d_interpolated = self.discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def _extract_discriminator_features(self, images):
        """Extract features from discriminator for feature matching."""
        with torch.no_grad():
            # Handle DataParallel wrapping
            if isinstance(self.discriminator, DataParallel):
                discriminator = self.discriminator.module
            else:
                discriminator = self.discriminator
            features = discriminator.conv_layers(images)
            features = features.view(features.size(0), -1)
        return features
    
    def _compute_regularization_loss(self):
        """Compute L1 and L2 regularization losses for encoder."""
        l1_loss = 0.0
        l2_loss = 0.0
        
        # Get encoder parameters (handle DataParallel)
        if isinstance(self.encoder, DataParallel):
            encoder_params = self.encoder.module.parameters()
        else:
            encoder_params = self.encoder.parameters()
        
        for param in encoder_params:
            if self.args.l1_reg > 0:
                l1_loss += torch.sum(torch.abs(param))
            if self.args.l2_reg > 0:
                l2_loss += torch.sum(param ** 2)
        
        total_reg_loss = 0.0
        if self.args.l1_reg > 0:
            total_reg_loss += self.args.l1_reg * l1_loss
        if self.args.l2_reg > 0:
            total_reg_loss += self.args.l2_reg * l2_loss
            
        return total_reg_loss
    
    def save_checkpoint(self, epoch):
        """Generate visualizations at checkpoint interval."""
        # Generate sample grids
        self.generate_sample_grids(epoch)
        
        # Analyze latent space
        self.analyze_latent_space(epoch)

        self.plot_loss_curves()
        
        print(f"Visualizations saved at epoch {epoch}")
    
    def generate_sample_grids(self, epoch):
        """Generate and save sample grids."""
        self.encoder.eval()
        self.generator.eval()
        
        with torch.no_grad():
            # Get a batch of samples
            batch = next(iter(self.dataloader))
            input_images, input_labels, actions, action_signs, target_labels = self._unpack_batch(batch)
            input_images = input_images.to(self.device)
            actions = actions.to(self.device)
            if action_signs is not None:
                action_signs = action_signs.to(self.device)
            target_labels = target_labels.to(self.device)
            
            # Generate fake images
            z = self._encode(input_images, actions, action_signs)
            fake_images = self.generator(z)
            
            # Create sample grid
            self._plot_sample_grid(
                input_images, input_labels, actions, target_labels, fake_images, epoch, action_signs
            )
        
        self.encoder.train()
        self.generator.train()
    
    def _plot_sample_grid(self, input_images, input_labels, actions, target_labels, fake_images, epoch, action_signs=None):
        """Plot sample grid with generated images only."""
        # Set font to avoid Times New Roman issues
        plt.rcParams['font.family'] = 'DejaVu Serif'
        
        batch_size = min(16, input_images.size(0))  # Show up to 16 samples
        
        fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 2, 2))
        if batch_size == 1:
            axes = [axes]
        
        for i in range(batch_size):
            # Get action value
            if self.args.image_actions:
                # Display action sign
                sign_str = '+' if action_signs[i].item() > 0 else '-'
                action_val = sign_str
            else:
                # Display action value from one-hot
                action_val = actions[i].argmax().item() - self.args.A
            
            # Generated image with title containing Input and Action
            axes[i].imshow(fake_images[i].cpu().squeeze(), cmap='gray')
            axes[i].set_title(f'Input: {input_labels[i].item()}\nAction: {action_val}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.samples_dir, f'samples_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def analyze_latent_space(self, epoch):
        """Analyze and visualize latent space."""
        self.encoder.eval()
        
        all_latents = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                input_images, input_labels, actions, action_signs, target_labels = self._unpack_batch(batch)
                input_images = input_images.to(self.device)
                if action_signs is not None:
                    action_signs = action_signs.to(self.device)
                
                # Extract latent vectors
                z = self._encode(input_images, actions, action_signs)
                actions = target_labels - input_labels
                all_latents.append(z[abs(actions)==0].cpu().numpy())
                all_targets.append(target_labels[abs(actions)==0].numpy())
        
        # Concatenate all latents and targets
        all_latents = np.concatenate(all_latents, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Sort by target label for consistent visualization
        sort_indices = np.argsort(all_targets)
        all_latents = all_latents[sort_indices]
        all_targets = all_targets[sort_indices]
        
        # Plot pairwise distance matrix
        self._plot_distance_matrix(all_latents, all_targets, epoch)
        
        # Plot class-mean distance matrix
        self._plot_class_mean_distance_matrix(all_latents, all_targets, epoch)
        
        # Plot PCA visualization
        # Save all_latents and all_targets to a file with A value and epoch
        save_path = os.path.join(self.plots_dir, f'latents_A{self.args.A}_epoch{epoch}.npz')
        np.savez(save_path, all_latents=all_latents, all_targets=all_targets)
        self._plot_pca_visualization(all_latents, all_targets, epoch)
        
        self.encoder.train()
    
    def _plot_distance_matrix(self, latents, targets, epoch):
        """Plot pairwise distance matrix of latent vectors."""
        # Set font to avoid Times New Roman issues
        plt.rcParams['font.family'] = 'DejaVu Serif'
        
        # Sample subset for visualization if too many points
        if len(latents) > 1000:
            indices = np.random.choice(len(latents), 1000, replace=False)
            latents_subset = latents[indices]
            targets_subset = targets[indices]
        else:
            latents_subset = latents
            targets_subset = targets
        
        # Compute pairwise distances
        distances = euclidean_distances(latents_subset)
        distances = distances[targets_subset.argsort(), :][:, targets_subset.argsort()]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(distances, cmap='viridis', square=True)
        plt.title(f'Pairwise Distance Matrix of Latent Vectors (Epoch {epoch})')
        plt.xlabel('Sample Index (sorted by target label)')
        plt.ylabel('Sample Index (sorted by target label)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'distance_matrix_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_class_mean_distance_matrix(self, latents, targets, epoch):
        """Plot distance matrix of class-mean latent vectors."""
        # Set font to avoid Times New Roman issues
        plt.rcParams['font.family'] = 'DejaVu Serif'
        
        # Compute class means
        class_means = []
        for digit in range(10):
            mask = targets == digit
            if np.any(mask):
                class_mean = np.mean(latents[mask], axis=0)
                class_means.append(class_mean)
            else:
                class_means.append(np.zeros(latents.shape[1]))
        
        class_means = np.array(class_means)
        
        # Compute pairwise distances between class means
        mean_distances = euclidean_distances(class_means)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_distances, 
                   xticklabels=range(10), 
                   yticklabels=range(10),
                   cmap='viridis', 
                   square=True,
                   annot=False, 
                   fmt='.2f')
        plt.title(f'Distance Matrix of Class-Mean Latent Vectors (Epoch {epoch})')
        plt.xlabel('Digit Class')
        plt.ylabel('Digit Class')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'class_mean_distance_matrix_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_visualization(self, latents, targets, epoch):
        """Plot PCA visualization of latent space."""
        # Set font to avoid Times New Roman issues
        plt.rcParams['font.family'] = 'DejaVu Serif'
        
        # Perform PCA
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        
        # Plot all samples
        plt.figure(figsize=(12, 5))
        
        # All samples colored by target label
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                            c=targets, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Label')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'PCA of All Latent Vectors (Epoch {epoch})')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Class means
        plt.subplot(1, 2, 2)
        class_means_2d = []
        for digit in range(10):
            mask = targets == digit
            if np.any(mask):
                class_mean = np.mean(latents_2d[mask], axis=0)
                class_means_2d.append(class_mean)
            else:
                class_means_2d.append([0, 0])
        
        class_means_2d = np.array(class_means_2d)
        scatter = plt.scatter(class_means_2d[:, 0], class_means_2d[:, 1], 
                            c=range(10), cmap='viridis', s=100, edgecolors='black')
        
        # Add labels
        for i, (x, y) in enumerate(class_means_2d):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.colorbar(scatter, label='Digit Class')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'PCA of Class-Mean Latent Vectors (Epoch {epoch})')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'pca_visualization_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def extract_first_pc(self):
        """Extract the first principal component from the current latent space."""
        self.encoder.eval()
        
        all_latents = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                input_images, input_labels, actions, action_signs, target_labels = self._unpack_batch(batch)
                input_images = input_images.to(self.device)
                actions = actions.to(self.device)
                if action_signs is not None:
                    action_signs = action_signs.to(self.device)
                
                # Extract latent vectors
                z = self._encode(input_images, actions, action_signs)
                all_latents.append(z.cpu().numpy())
        
        # Concatenate all latents
        all_latents = np.concatenate(all_latents, axis=0)
        
        # Compute PCA and extract first component
        pca = PCA(n_components=1)
        first_pc = pca.fit_transform(all_latents).flatten()
        
        # Get the first principal component vector
        first_pc_vector = pca.components_[0]  # Shape: (latent_dim,)
        
        # Compute mean of latent space
        latent_mean = np.mean(all_latents, axis=0)
        
        self.encoder.train()
        
        return first_pc_vector, latent_mean, first_pc.min(), first_pc.max()
    
    def generate_pc_samples(self, epoch):
        """Generate samples along the first principal component."""
        print(f"Generating PC samples for epoch {epoch}...")
        
        # Extract first PC information
        first_pc_vector, latent_mean, pc_min, pc_max = self.extract_first_pc()
        
        # Generate samples along the PC
        pc_values = np.linspace(pc_min, pc_max, self.args.samples_gen_pc)
        
        self.encoder.eval()
        self.generator.eval()
        
        generated_images = []
        
        with torch.no_grad():
            for pc_val in pc_values:
                # Create latent vector by moving along first PC
                latent_vector = latent_mean + pc_val * first_pc_vector
                latent_tensor = torch.FloatTensor(latent_vector).unsqueeze(0).to(self.device)
                
                # Generate image
                generated_image = self.generator(latent_tensor)
                generated_images.append(generated_image.cpu())
        
        # Convert to tensor
        generated_images = torch.cat(generated_images, dim=0)
        
        # Plot and save results
        self._plot_pc_samples(generated_images, pc_values, epoch)
        
        self.encoder.train()
        self.generator.train()
        
        print(f"PC samples saved for epoch {epoch}")
    
    def _plot_pc_samples(self, generated_images, pc_values, epoch):
        """Plot generated images along the first principal component."""
        # Set font to avoid Times New Roman issues
        plt.rcParams['font.family'] = 'DejaVu Serif'
        
        num_samples = len(generated_images)
        
        # Create grid layout
        cols = min(10, num_samples)  # Max 10 columns
        rows = (num_samples + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            row = i // cols
            col = i % cols
            
            axes[row, col].imshow(generated_images[i].squeeze(), cmap='gray')
            axes[row, col].set_title(f'PC: {pc_values[i]:.3f}', fontname='Times New Roman')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        # plt.suptitle(
        #     f'Images Generated Along First Principal Component',
        #     fontsize=16,
        #     fontname='Times New Roman'
        # )
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'pc_samples_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_loss_curves(self):
        """Plot and save loss curves on log scale."""
        print("Plotting loss curves...")
        
        # Set font to avoid Times New Roman issues
        plt.rcParams['font.family'] = 'DejaVu Serif'
        
        epochs = range(1, len(self.loss_history['d_loss']) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Loss Curves (Log Scale)', fontsize=16)
        
        # Plot 1: Discriminator vs Generator Loss
        axes[0, 0].semilogy(epochs, self.loss_history['d_loss'], 'b-', label='Discriminator Loss', linewidth=2)
        axes[0, 0].semilogy(epochs, self.loss_history['g_loss'], 'r-', label='Generator Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (log scale)')
        axes[0, 0].set_title('Discriminator vs Generator Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Real vs Fake Loss
        axes[0, 1].semilogy(epochs, self.loss_history['real_loss'], 'g-', label='Real Loss', linewidth=2)
        axes[0, 1].semilogy(epochs, self.loss_history['fake_loss'], 'm-', label='Fake Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (log scale)')
        axes[0, 1].set_title('Real vs Fake Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: All Losses Combined
        axes[1, 0].semilogy(epochs, self.loss_history['d_loss'], 'b-', label='Discriminator', linewidth=2)
        axes[1, 0].semilogy(epochs, self.loss_history['g_loss'], 'r-', label='Generator', linewidth=2)
        axes[1, 0].semilogy(epochs, self.loss_history['real_loss'], 'g-', label='Real', linewidth=2)
        axes[1, 0].semilogy(epochs, self.loss_history['fake_loss'], 'm-', label='Fake', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (log scale)')
        axes[1, 0].set_title('All Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Reconstruction Loss (if used)
        if any(loss > 0 for loss in self.loss_history['recon_loss']):
            axes[1, 1].semilogy(epochs, self.loss_history['recon_loss'], 'c-', label='Reconstruction Loss', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss (log scale)')
            axes[1, 1].set_title('Reconstruction Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # If no reconstruction loss, show discriminator loss ratio
            d_g_ratio = [d/g if g > 0 else 0 for d, g in zip(self.loss_history['d_loss'], self.loss_history['g_loss'])]
            axes[1, 1].plot(epochs, d_g_ratio, 'k-', label='D/G Ratio', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Ratio')
            axes[1, 1].set_title('Discriminator/Generator Loss Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'loss_curves_log_scale.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create a single combined plot
        plt.figure(figsize=(12, 8))
        plt.semilogy(epochs, self.loss_history['d_loss'], 'b-', label='Discriminator Loss', linewidth=2)
        plt.semilogy(epochs, self.loss_history['g_loss'], 'r-', label='Generator Loss', linewidth=2)
        plt.semilogy(epochs, self.loss_history['real_loss'], 'g-', label='Real Loss', linewidth=2)
        plt.semilogy(epochs, self.loss_history['fake_loss'], 'm-', label='Fake Loss', linewidth=2)
        
        if any(loss > 0 for loss in self.loss_history['recon_loss']):
            plt.semilogy(epochs, self.loss_history['recon_loss'], 'c-', label='Reconstruction Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss Curves (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'loss_curves_combined_log_scale.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Loss curves saved successfully!")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.args.epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            epoch_real_loss = 0
            epoch_fake_loss = 0
            epoch_recon_loss = 0
            
            for batch_idx, batch in enumerate(self.dataloader):
                input_images, input_labels, actions, action_signs, target_labels = self._unpack_batch(batch)
                input_images = input_images.to(self.device)
                input_labels = input_labels.to(self.device)
                actions = actions.to(self.device)
                if action_signs is not None:
                    action_signs = action_signs.to(self.device)
                target_labels = target_labels.to(self.device)
                
                # Determine discriminator training ratio
                if self.args.adaptive_d_train_ratio and len(self.loss_history['d_loss']) > 0:
                    # Use recent losses to determine training ratio
                    recent_d_loss = np.mean(self.loss_history['d_loss'][-5:]) if len(self.loss_history['d_loss']) >= 5 else self.loss_history['d_loss'][-1]
                    recent_g_loss = np.mean(self.loss_history['g_loss'][-5:]) if len(self.loss_history['g_loss']) >= 5 else self.loss_history['g_loss'][-1]
                    d_train_ratio = self.get_adaptive_d_train_ratio(recent_d_loss, recent_g_loss)
                else:
                    d_train_ratio = self.args.d_train_ratio
                
                # Train discriminator multiple times per generator update
                for _ in range(d_train_ratio):
                    with torch.no_grad():
                        z = self._encode(input_images, actions, action_signs)
                        fake_images = self.generator(z)
                    
                    d_loss, real_loss, fake_loss = self.train_discriminator(
                        input_images, input_labels, fake_images, target_labels
                    )
                
                # Train generator and encoder
                g_loss, gan_loss, recon_loss = self.train_generator_encoder(
                    input_images, input_labels, actions, target_labels, action_signs
                )
                
                # Accumulate losses
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                epoch_real_loss += real_loss
                epoch_fake_loss += fake_loss
                epoch_recon_loss += recon_loss
            
            # Average losses
            num_batches = len(self.dataloader)
            epoch_d_loss /= num_batches
            epoch_g_loss /= num_batches
            epoch_real_loss /= num_batches
            epoch_fake_loss /= num_batches
            epoch_recon_loss /= num_batches
            
            # Store loss history
            self.loss_history['d_loss'].append(epoch_d_loss)
            self.loss_history['g_loss'].append(epoch_g_loss)
            self.loss_history['real_loss'].append(epoch_real_loss)
            self.loss_history['fake_loss'].append(epoch_fake_loss)
            self.loss_history['recon_loss'].append(epoch_recon_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.args.epochs}: "
                  f"D_loss: {epoch_d_loss:.4f}, G_loss: {epoch_g_loss:.4f}, "
                  f"Real_loss: {epoch_real_loss:.4f}, Fake_loss: {epoch_fake_loss:.4f}, "
                  f"Recon_loss: {epoch_recon_loss:.4f}")
            
            # Check training stability
            self.check_training_stability(epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % self.args.checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)
            
            # Generate PC samples
            if (epoch + 1) % self.args.pc_sampling_interval == 0:
                self.generate_pc_samples(epoch + 1)
            
            # Update learning rates
            if self.scheduler_G is not None:
                self.scheduler_G.step(epoch_g_loss)
            if self.scheduler_D is not None:
                self.scheduler_D.step(epoch_d_loss)
        
        print("Training completed!")
        
        # Plot loss curves at the end of training
        self.plot_loss_curves()
    
    def run_pc_sampling(self, epoch=None):
        """Standalone function to run PC sampling on current model."""
        if epoch is None:
            epoch = self.args.epochs
        print(f"Running PC sampling for epoch {epoch}...")
        self.generate_pc_samples(epoch)
    
    def run_all_plots(self, epoch=None):
        """Run all plotting functions on current model."""
        if epoch is None:
            epoch = self.args.epochs
        print(f"Running all plotting functions for epoch {epoch}...")
        
        # Generate sample grids
        self.generate_sample_grids(epoch)
        
        # Analyze latent space
        self.analyze_latent_space(epoch)
        
        # Generate PC samples
        self.generate_pc_samples(epoch)
        
        # Plot loss curves (if loss history exists)
        if self.loss_history['d_loss']:
            self.plot_loss_curves()
        
        print("All plotting functions completed!")
    
    def get_adaptive_d_train_ratio(self, d_loss, g_loss):
        """Calculate adaptive discriminator training ratio based on loss balance."""
        if g_loss == 0:
            return 1
        
        current_ratio = d_loss / g_loss
        target_ratio = self.args.target_d_g_ratio
        
        if current_ratio < target_ratio * 0.5:
            # Discriminator is too weak, train it more
            return min(5, self.args.d_train_ratio * 2)
        elif current_ratio > target_ratio * 2:
            # Discriminator is too strong, train it less
            return max(1, self.args.d_train_ratio // 2)
        else:
            return self.args.d_train_ratio
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint, handling DataParallel models."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle DataParallel models
        encoder_state = checkpoint['encoder_state_dict']
        generator_state = checkpoint['generator_state_dict']
        discriminator_state = checkpoint['discriminator_state_dict']
        
        # Check if current models are wrapped with DataParallel
        current_encoder_is_parallel = isinstance(self.encoder, DataParallel)
        current_generator_is_parallel = isinstance(self.generator, DataParallel)
        current_discriminator_is_parallel = isinstance(self.discriminator, DataParallel)
        
        # Check if checkpoint was saved with DataParallel
        checkpoint_encoder_is_parallel = any(key.startswith('module.') for key in encoder_state.keys())
        checkpoint_generator_is_parallel = any(key.startswith('module.') for key in generator_state.keys())
        checkpoint_discriminator_is_parallel = any(key.startswith('module.') for key in discriminator_state.keys())
        
        # Adjust state dict keys based on current model architecture
        if checkpoint_encoder_is_parallel and not current_encoder_is_parallel:
            # Remove 'module.' prefix
            encoder_state = {k[7:]: v for k, v in encoder_state.items()}
        elif not checkpoint_encoder_is_parallel and current_encoder_is_parallel:
            # Add 'module.' prefix
            encoder_state = {f'module.{k}': v for k, v in encoder_state.items()}
            
        if checkpoint_generator_is_parallel and not current_generator_is_parallel:
            # Remove 'module.' prefix
            generator_state = {k[7:]: v for k, v in generator_state.items()}
        elif not checkpoint_generator_is_parallel and current_generator_is_parallel:
            # Add 'module.' prefix
            generator_state = {f'module.{k}': v for k, v in generator_state.items()}
            
        if checkpoint_discriminator_is_parallel and not current_discriminator_is_parallel:
            # Remove 'module.' prefix
            discriminator_state = {k[7:]: v for k, v in discriminator_state.items()}
        elif not checkpoint_discriminator_is_parallel and current_discriminator_is_parallel:
            # Add 'module.' prefix
            discriminator_state = {f'module.{k}': v for k, v in discriminator_state.items()}
        
        self.encoder.load_state_dict(encoder_state)
        self.generator.load_state_dict(generator_state)
        self.discriminator.load_state_dict(discriminator_state)
        
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
    
    def check_training_stability(self, epoch):
        """Check training stability and suggest adjustments."""
        if len(self.loss_history['d_loss']) < 10:
            return
        
        # Check for increasing losses
        recent_d_losses = self.loss_history['d_loss'][-10:]
        recent_g_losses = self.loss_history['g_loss'][-10:]
        
        d_trend = np.polyfit(range(len(recent_d_losses)), recent_d_losses, 1)[0]
        g_trend = np.polyfit(range(len(recent_g_losses)), recent_g_losses, 1)[0]
        
        # Check for mode collapse (discriminator loss very low)
        if recent_d_losses[-1] < 0.1:
            print(f"Warning: Discriminator loss very low ({recent_d_losses[-1]:.4f}) - possible mode collapse")
        
        # Check for discriminator overpowering generator
        if recent_d_losses[-1] < recent_g_losses[-1] * 0.1:
            print(f"Warning: Discriminator much stronger than generator (D: {recent_d_losses[-1]:.4f}, G: {recent_g_losses[-1]:.4f})")
        
        # Check for both losses increasing
        if d_trend > 0 and g_trend > 0:
            print(f"Warning: Both discriminator and generator losses increasing (D trend: {d_trend:.4f}, G trend: {g_trend:.4f})")
        
        # Check for oscillation
        d_std = np.std(recent_d_losses)
        g_std = np.std(recent_g_losses)
        if d_std > np.mean(recent_d_losses) * 0.5 or g_std > np.mean(recent_g_losses) * 0.5:
            print(f"Warning: High loss variance detected (D std: {d_std:.4f}, G std: {g_std:.4f})")


def save_command_line_args(args, parser, output_dir):
    """
    Save all command line arguments (including defaults) to a text file in command line format.
    This ensures reproducibility by capturing all simulation parameters.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all arguments as a dictionary
    args_dict = vars(args)
    
    # Get default values from parser actions
    parser_defaults = {}
    for action in parser._actions:
        if action.dest != 'help' and hasattr(action, 'default'):
            parser_defaults[action.dest] = action.default
    
    # Build command line string
    cmd_parts = ['python', 'small_MNIST.py']
    
    # Build command with all arguments, including defaults
    # This ensures full reproducibility by explicitly showing all parameter values
    for arg_name, arg_value in sorted(args_dict.items()):
        # Skip None values (they won't be in the command)
        if arg_value is None:
            continue
        
        # Convert argument name to command line format (with --)
        arg_flag = '--' + arg_name
        
        # Handle boolean flags
        if isinstance(arg_value, bool):
            # For action='store_true', only include if True
            # For type=bool, include if True (False is usually default and not needed)
            # However, to show all arguments including defaults, we could include False too,
            # but argparse doesn't support --flag False syntax for store_true actions
            if arg_value:
                cmd_parts.append(arg_flag)
            # Note: For False values with action='store_true', the flag simply isn't included
            # This is the correct command-line representation
        else:
            # Add flag and value for non-boolean arguments (including defaults)
            cmd_parts.extend([arg_flag, str(arg_value)])
    
    # Join command parts
    complete_cmd = ' '.join(cmd_parts)
    
    # Save to file
    cmd_file = os.path.join(output_dir, 'command_line.txt')
    with open(cmd_file, 'w') as f:
        f.write("# Complete command line with all arguments (including defaults)\n")
        f.write("# Generated automatically by small_MNIST.py\n")
        f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(complete_cmd + "\n")
        f.write("\n# All arguments with values:\n")
        for arg_name, arg_value in sorted(args_dict.items()):
            default_val = parser_defaults.get(arg_name, 'N/A')
            f.write(f"#   {arg_name}: {arg_value} (default: {default_val})\n")
    
    print(f"Command line arguments saved to: {cmd_file}")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Train MNIST Action GAN')
    
    # Dataset parameters
    parser.add_argument('--N', type=int, default=20, 
                       help='Number of samples per digit class (default: 10)')
    parser.add_argument('--A', type=int, default=5, 
                       help='Action range [-A, A] (default: 2)')
    parser.add_argument('--cyclic', type=bool, default=False, 
                       help='Use cyclic addition mod 10 (default: False)')
    parser.add_argument('--image_actions', action='store_true',
                       help='Use MNIST images as actions instead of scalar values (default: False)')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=64, 
                       help='Latent dimension (default: 64)')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Batch size (default: 64)')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=2e-4, 
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of epochs (default: 100)')
    parser.add_argument('--checkpoint_interval', type=int, default=10, 
                       help='Checkpoint interval (default: 10)')
    parser.add_argument('--samples_M', type=int, default=256, 
                       help='Number of samples for analysis (default: 256)')
    parser.add_argument('--lambda_recon', type=float, default=0.0, 
                       help='Reconstruction loss weight (default: 0.0)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed (default: 42)')
    parser.add_argument('--n_layers', type=int, default=2, 
                       help='Number of layers for combined encoder (default: 2)')
    
    # PC sampling parameters
    parser.add_argument('--samples_gen_pc', type=int, default=10, 
                       help='Number of samples to generate along first PC (default: 20)')
    parser.add_argument('--pc_sampling_interval', type=int, default=20, 
                       help='Interval for PC sampling during training (default: 20)')
    parser.add_argument('--plot_only', action='store_true', 
                       help='Run only plotting functions without training (requires existing checkpoint)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint for plot-only mode')
    parser.add_argument('--run_directory', type=str, default='default',
                       help='Directory name for organizing results (default: default)')
    
    # Training stabilization parameters
    parser.add_argument('--use_lr_scheduler', action='store_true',
                       help='Use learning rate scheduler for generator')
    parser.add_argument('--gradient_clip', type=float, default=0.0,
                       help='Gradient clipping threshold (0 = no clipping)')
    parser.add_argument('--d_loss_weight', type=float, default=1.0,
                       help='Discriminator loss weight (default: 1.0)')
    parser.add_argument('--g_loss_weight', type=float, default=1.0,
                       help='Generator loss weight (default: 1.0)')
    parser.add_argument('--use_feature_matching', action='store_true',
                       help='Use feature matching loss for generator')
    parser.add_argument('--use_gradient_penalty', action='store_true',
                       help='Use gradient penalty for discriminator')
    parser.add_argument('--gp_weight', type=float, default=10.0,
                       help='Gradient penalty weight (default: 10.0)')
    parser.add_argument('--d_train_ratio', type=int, default=1,
                       help='Number of discriminator updates per generator update (default: 1)')
    
    # Additional training balance parameters
    parser.add_argument('--adversarial_weight', type=float, default=0.1,
                       help='Weight for adversarial loss in generator (default: 0.1)')
    parser.add_argument('--feature_matching_weight', type=float, default=1.0,
                       help='Weight for feature matching loss (default: 1.0)')
    parser.add_argument('--d_lr', type=float, default=None,
                       help='Discriminator learning rate (default: same as generator)')
    parser.add_argument('--g_lr', type=float, default=None,
                       help='Generator learning rate (default: same as discriminator)')
    parser.add_argument('--adaptive_d_train_ratio', action='store_true',
                       help='Adaptively adjust discriminator training ratio based on loss balance')
    parser.add_argument('--target_d_g_ratio', type=float, default=1.0,
                       help='Target discriminator/generator loss ratio for adaptive training (default: 1.0)')
    
    # Regularization parameters
    parser.add_argument('--l1_reg', type=float, default=0.0,
                       help='L1 regularization weight for encoder (default: 0.0)')
    parser.add_argument('--l2_reg', type=float, default=0.0,
                       help='L2 regularization weight for encoder (default: 0.0)')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                       help='Dropout rate for encoder (default: 0.0)')
    parser.add_argument('--regularization_type', type=str, default='none',
                       choices=['none', 'l1', 'l2', 'dropout', 'l1_l2', 'all'],
                       help='Type of regularization to apply (default: none)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # Create base directory and save command line arguments
    base_dir = f"MNIST_small/{args.run_directory}"
    os.makedirs(base_dir, exist_ok=True)
    save_command_line_args(args, parser, base_dir)
    
    # Create model
    model = MNISTActionGAN(args)
    
    if args.plot_only:
        # Load checkpoint if specified
        if args.checkpoint_path:
            if os.path.exists(args.checkpoint_path):
                model.load_checkpoint(args.checkpoint_path)
            else:
                print(f"Checkpoint not found at {args.checkpoint_path}")
                return
        else:
            # Find the latest checkpoint in the specified run directory
            if os.path.exists(model.checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(model.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint_path = os.path.join(model.checkpoint_dir, latest_checkpoint)
                    model.load_checkpoint(checkpoint_path)
                else:
                    print(f"No checkpoints found in {model.checkpoint_dir}. Please train the model first or specify a checkpoint path.")
                    return
            else:
                print(f"Checkpoint directory {model.checkpoint_dir} does not exist. Please train the model first or specify a checkpoint path.")
                return
        
        # Run all plotting functions
        model.run_all_plots()
    else:
        # Train model
        model.train()


if __name__ == '__main__':
    main()
