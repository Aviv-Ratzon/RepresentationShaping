"""
Standalone PredNet Training on Moving MNIST

This script implements a modern PredNet architecture for video prediction on Moving MNIST.
PredNet uses hierarchical prediction errors to learn temporal representations.

Based on: Lotter et al. (2016) "Deep Predictive Coding Networks for Video Prediction"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class MovingMNISTDataset(Dataset):
    """Dataset for Moving MNIST sequences."""
    
    def __init__(self, num_digits=2, image_size=64, num_frames=10, 
                 digit_size=28, step_length=1, max_velocity=4, 
                 num_samples=10000, train=True, seed=42):
        """
        Initialize Moving MNIST dataset.
        
        Args:
            num_digits: Number of digits in each sequence
            image_size: Size of output images (square)
            num_frames: Number of frames per sequence
            digit_size: Size of MNIST digits (28x28)
            step_length: Number of frames to predict ahead
            max_velocity: Maximum pixel velocity per frame
            num_samples: Number of sequences to generate
            train: Whether to use training or test MNIST data
            seed: Random seed
        """
        self.num_digits = num_digits
        self.image_size = image_size
        self.num_frames = num_frames
        self.digit_size = digit_size
        self.step_length = step_length
        self.max_velocity = max_velocity
        self.num_samples = num_samples
        
        # Load MNIST dataset
        mnist_dataset = MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((digit_size, digit_size)),
                transforms.ToTensor()
            ])
        )
        
        self.mnist_data = []
        self.mnist_labels = []
        for i in range(len(mnist_dataset)):
            img, label = mnist_dataset[i]
            self.mnist_data.append(img.squeeze().numpy())
            self.mnist_labels.append(label)
        
        self.mnist_data = np.array(self.mnist_data)
        self.mnist_labels = np.array(self.mnist_labels)
        
        # Generate sequences
        np.random.seed(seed)
        self.sequences = self._generate_sequences()
        
    def _generate_sequences(self):
        """Generate Moving MNIST sequences."""
        sequences = []
        
        for _ in tqdm(range(self.num_samples), desc="Generating sequences"):
            # Randomly select digits
            digit_indices = np.random.choice(
                len(self.mnist_data), 
                size=self.num_digits, 
                replace=False
            )
            digits = [self.mnist_data[idx] for idx in digit_indices]
            
            # Initialize positions and velocities
            positions = []
            velocities = []
            for _ in range(self.num_digits):
                x = np.random.randint(0, self.image_size - self.digit_size)
                y = np.random.randint(0, self.image_size - self.digit_size)
                vx = np.random.uniform(-self.max_velocity, self.max_velocity)
                vy = np.random.uniform(-self.max_velocity, self.max_velocity)
                positions.append([x, y])
                velocities.append([vx, vy])
            
            # Generate sequence
            sequence = []
            for frame_idx in range(self.num_frames):
                canvas = np.zeros((self.image_size, self.image_size))
                
                for digit_idx in range(self.num_digits):
                    x, y = positions[digit_idx]
                    vx, vy = velocities[digit_idx]
                    
                    # Place digit on canvas
                    x_int = int(x)
                    y_int = int(y)
                    if 0 <= x_int < self.image_size - self.digit_size and \
                       0 <= y_int < self.image_size - self.digit_size:
                        canvas[y_int:y_int+self.digit_size, 
                               x_int:x_int+self.digit_size] = np.maximum(
                            canvas[y_int:y_int+self.digit_size, 
                                  x_int:x_int+self.digit_size],
                            digits[digit_idx]
                        )
                    
                    # Update position
                    positions[digit_idx][0] += velocities[digit_idx][0]
                    positions[digit_idx][1] += velocities[digit_idx][1]
                    
                    # Bounce off walls
                    if positions[digit_idx][0] <= 0 or \
                       positions[digit_idx][0] >= self.image_size - self.digit_size:
                        velocities[digit_idx][0] *= -1
                    if positions[digit_idx][1] <= 0 or \
                       positions[digit_idx][1] >= self.image_size - self.digit_size:
                        velocities[digit_idx][1] *= -1
                    
                    # Clamp positions
                    positions[digit_idx][0] = np.clip(
                        positions[digit_idx][0], 
                        0, 
                        self.image_size - self.digit_size
                    )
                    positions[digit_idx][1] = np.clip(
                        positions[digit_idx][1], 
                        0, 
                        self.image_size - self.digit_size
                    )
                
                sequence.append(canvas.copy())
            
            sequences.append(np.array(sequence))
        
        return np.array(sequences)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Normalize to [0, 1]
        sequence = sequence.astype(np.float32) / 255.0
        
        # Input: first num_frames - step_length frames
        # Target: last step_length frames
        input_frames = sequence[:self.num_frames - self.step_length]
        target_frames = sequence[self.num_frames - self.step_length:]
        
        # Convert to tensors: [T, H, W] -> [T, 1, H, W]
        input_frames = torch.FloatTensor(input_frames).unsqueeze(1)
        target_frames = torch.FloatTensor(target_frames).unsqueeze(1)
        
        return input_frames, target_frames


class PredNetLayer(nn.Module):
    """A single layer of PredNet with LSTM-like recurrent connections."""
    
    def __init__(self, R_channels, A_channels):
        """
        Initialize PredNet layer.
        
        Args:
            R_channels: Number of channels in representation (hidden state)
            A_channels: Number of channels in input/error
        """
        super(PredNetLayer, self).__init__()
        
        self.R_channels = R_channels
        self.A_channels = A_channels
        
        # Bottom-up: error -> representation update
        self.conv_r = nn.Conv2d(
            A_channels + R_channels,  # Input: error + previous representation
            R_channels, 
            kernel_size=3, 
            padding=1
        )
        
        # Top-down: representation -> prediction
        self.conv_p = nn.Conv2d(
            R_channels, 
            A_channels, 
            kernel_size=3, 
            padding=1
        )
        
        # Recurrent state
        self.r = None
        
    def init_hidden(self, batch_size, height, width, device):
        """Initialize hidden state."""
        self.r = torch.zeros(
            batch_size, 
            self.R_channels, 
            height, 
            width,
            device=device
        )
    
    def forward(self, a, r_up=None):
        """
        Forward pass.
        
        Args:
            a: Input/error from layer below [B, A_channels, H, W]
            r_up: Representation from layer above (for top-down prediction)
        
        Returns:
            error: Prediction error [B, A_channels, H, W]
            r: Updated representation [B, R_channels, H, W]
        """
        batch_size, _, height, width = a.shape
        device = a.device
        
        if self.r is None:
            self.init_hidden(batch_size, height, width, device)
        
        # Combine error with previous representation
        if r_up is not None:
            # If we have representation from above, use it for prediction
            # (This would be for top-down pass, simplified here)
            pass
        
        # Update representation: combine error and previous representation
        r_input = torch.cat([a, self.r], dim=1)  # [B, A_channels + R_channels, H, W]
        r_update = torch.relu(self.conv_r(r_input))
        
        # Update recurrent state (simple update, can be made more sophisticated)
        self.r = r_update
        
        # Generate prediction from representation
        p = torch.sigmoid(self.conv_p(self.r))
        
        # Compute prediction error
        error = a - p
        
        return error, self.r


class PredNet(nn.Module):
    """PredNet architecture for video prediction."""
    
    def __init__(self, input_shape=(64, 64), stack_sizes=(1, 48, 96, 192),
                 R_stack_sizes=(1, 48, 96, 192), layer_loss_weights=(1., 0.1, 0.1, 0.1)):
        """
        Initialize PredNet.
        
        Args:
            input_shape: (height, width) of input images
            stack_sizes: Number of channels in each layer (A channels)
            R_stack_sizes: Number of channels in representation (R channels)
            layer_loss_weights: Weight for loss at each layer
        """
        super(PredNet, self).__init__()
        
        self.input_shape = input_shape
        self.stack_sizes = stack_sizes
        self.R_stack_sizes = R_stack_sizes
        self.nb_layers = len(stack_sizes)
        self.layer_loss_weights = layer_loss_weights
        
        # Build layers
        self.layers = nn.ModuleList()
        # Build channel transformation layers between levels
        self.error_upsample = nn.ModuleList()
        
        for i in range(self.nb_layers):
            R_channels = R_stack_sizes[i]
            A_channels = stack_sizes[i]
            
            layer = PredNetLayer(R_channels, A_channels)
            self.layers.append(layer)
            
            # Add upsampling layer to transform error from layer i to layer i+1
            if i < self.nb_layers - 1:
                next_A_channels = stack_sizes[i + 1]
                upsample = nn.Conv2d(
                    A_channels, 
                    next_A_channels, 
                    kernel_size=3, 
                    padding=1,
                    stride=1
                )
                self.error_upsample.append(upsample)
            else:
                self.error_upsample.append(None)
        
    def reset_states(self):
        """Reset all layer states."""
        for layer in self.layers:
            layer.r = None
    
    def forward(self, input_sequence):
        """
        Forward pass.
        
        Args:
            input_sequence: Input frames [B, T, C, H, W]
        
        Returns:
            predictions: Predicted frames [B, T_pred, C, H, W]
            errors: Prediction errors at each layer and time step
        """
        batch_size, num_frames, channels, height, width = input_sequence.shape
        
        # Reset states
        self.reset_states()
        
        all_errors = []
        all_predictions = []
        
        # Process input sequence
        for t in range(num_frames):
            a = input_sequence[:, t]  # [B, C, H, W]
            errors = []
            
            # Bottom-up pass through layers
            current_a = a
            for i, layer in enumerate(self.layers):
                error, r = layer(current_a)
                errors.append(error)
                # Transform error to match next layer's input channels
                if i < self.nb_layers - 1 and self.error_upsample[i] is not None:
                    current_a = self.error_upsample[i](error)  # Transform error for next layer
                else:
                    current_a = error  # Last layer, no transformation needed
            
            # Store errors for loss computation
            all_errors.append(errors)
            
            # Generate prediction from bottom layer's representation
            # Use the first layer's prediction as output
            pred = torch.sigmoid(self.layers[0].conv_p(self.layers[0].r))
            all_predictions.append(pred)
        
        # For prediction, use the last frame's prediction
        # In practice, we predict the next frame after seeing input sequence
        predictions = all_predictions[-1].unsqueeze(1)  # [B, 1, C, H, W]
        
        return predictions, all_errors
    
    def compute_loss(self, predictions, targets, errors):
        """
        Compute PredNet loss (weighted sum of prediction errors).
        
        Args:
            predictions: Predicted frames [B, T, C, H, W]
            targets: Target frames [B, T, C, H, W]
            errors: List of errors at each layer and time step
        """
        # Reconstruction loss (MSE between predictions and targets)
        # Take first target frame for single-step prediction
        if targets.size(1) > 1:
            target = targets[:, 0:1]  # [B, 1, C, H, W]
        else:
            target = targets
        
        recon_loss = nn.functional.mse_loss(predictions, target)
        
        # Prediction error loss (sum of errors at each layer)
        # Use errors from the last time step
        error_loss = 0.0
        if len(errors) > 0:
            last_errors = errors[-1]  # Errors from last time step
            for i, error in enumerate(last_errors):
                if i < len(self.layer_loss_weights) and self.layer_loss_weights[i] > 0:
                    error_loss += self.layer_loss_weights[i] * torch.mean(error ** 2)
        
        total_loss = recon_loss + 0.1 * error_loss
        return total_loss, recon_loss, error_loss


def train_prednet(model, train_loader, val_loader, num_epochs=50, 
                  learning_rate=0.001, device='cpu', save_dir='./checkpoints'):
    """
    Train PredNet model.
    
    Args:
        model: PredNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_error_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, targets in pbar:
            inputs = inputs.to(device)  # [B, T_in, C, H, W]
            targets = targets.to(device)  # [B, T_out, C, H, W]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, errors = model(inputs)
            
            # Compute loss
            loss, recon_loss, error_loss = model.compute_loss(
                predictions, targets, errors
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_error_loss += error_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'error': f'{error_loss.item():.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_error_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                predictions, errors = model(inputs)
                loss, recon_loss, error_loss = model.compute_loss(
                    predictions, targets, errors
                )
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_error_loss += error_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f} (Recon: {train_recon_loss/len(train_loader):.4f}, '
              f'Error: {train_error_loss/len(train_loader):.4f})')
        print(f'  Val Loss: {avg_val_loss:.4f} (Recon: {val_recon_loss/len(val_loader):.4f}, '
              f'Error: {val_error_loss/len(val_loader):.4f})')
        print()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'best_prednet.pth'))
            print(f'  Saved best model (val_loss: {avg_val_loss:.4f})')
            print()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def plot_training_curves(history):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    ax.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('PredNet Training Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prednet_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to 'prednet_training_curves.png'")
    plt.show()


def visualize_predictions(model, data_loader, device='cpu', num_examples=5):
    """
    Visualize prediction examples.
    
    Args:
        model: Trained PredNet model
        data_loader: DataLoader for visualization
        device: Device to run on
        num_examples: Number of examples to visualize
    """
    model.eval()
    
    with torch.no_grad():
        inputs, targets = next(iter(data_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        predictions, _ = model(inputs)
        
        # Select random examples
        indices = np.random.choice(inputs.size(0), num_examples, replace=False)
        
        fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            # Last input frame
            last_input = inputs[idx, -1, 0].cpu().numpy()
            # First target frame
            target = targets[idx, 0, 0].cpu().numpy()
            # First prediction
            pred = predictions[idx, 0, 0].cpu().numpy()
            
            axes[i, 0].imshow(last_input, cmap='gray')
            axes[i, 0].set_title('Last Input Frame', fontsize=10)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target, cmap='gray')
            axes[i, 1].set_title('Ground Truth', fontsize=10)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction', fontsize=10)
            axes[i, 2].axis('off')
        
        plt.suptitle('PredNet Predictions on Moving MNIST', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('prednet_predictions.png', dpi=150, bbox_inches='tight')
        print("Predictions saved to 'prednet_predictions.png'")
        plt.show()


def main():
    """Main function to run PredNet training."""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configuration
    config = {
        'num_digits': 2,
        'image_size': 64,
        'num_frames': 10,
        'digit_size': 28,
        'step_length': 1,
        'max_velocity': 4,
        'num_train_samples': 8000,
        'num_val_samples': 1000,
        'num_test_samples': 1000,
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 70)
    print("PredNet Training on Moving MNIST")
    print("=" * 70)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create datasets
    print("Creating Moving MNIST datasets...")
    train_dataset = MovingMNISTDataset(
        num_digits=config['num_digits'],
        image_size=config['image_size'],
        num_frames=config['num_frames'],
        digit_size=config['digit_size'],
        step_length=config['step_length'],
        max_velocity=config['max_velocity'],
        num_samples=config['num_train_samples'],
        train=True,
        seed=42
    )
    
    val_dataset = MovingMNISTDataset(
        num_digits=config['num_digits'],
        image_size=config['image_size'],
        num_frames=config['num_frames'],
        digit_size=config['digit_size'],
        step_length=config['step_length'],
        max_velocity=config['max_velocity'],
        num_samples=config['num_val_samples'],
        train=True,
        seed=43
    )
    
    test_dataset = MovingMNISTDataset(
        num_digits=config['num_digits'],
        image_size=config['image_size'],
        num_frames=config['num_frames'],
        digit_size=config['digit_size'],
        step_length=config['step_length'],
        max_velocity=config['max_velocity'],
        num_samples=config['num_test_samples'],
        train=False,
        seed=44
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    # Create model
    print("Creating PredNet model...")
    model = PredNet(
        input_shape=(config['image_size'], config['image_size']),
        stack_sizes=(1, 48, 96, 192),
        R_stack_sizes=(1, 48, 96, 192),
        layer_loss_weights=(1., 0.1, 0.1, 0.1)
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Train model
    print("Starting training...")
    history = train_prednet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device']
    )
    
    print("\nTraining completed!")
    print(f"Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"Final val loss: {history['val_losses'][-1]:.4f}")
    print()
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_curves(history)
    
    # Load best model for visualization
    checkpoint = torch.load('./checkpoints/best_prednet.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model (val_loss: {checkpoint['val_loss']:.4f})")
    print()
    
    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(model, test_loader, device=config['device'], num_examples=5)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

