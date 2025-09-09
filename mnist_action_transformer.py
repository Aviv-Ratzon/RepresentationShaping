"""
MNIST Action Transformer Network

This module implements a neural network that takes an MNIST digit image and a one-hot encoded action
in the range [-max_move, max_move] and outputs an MNIST digit from the class corresponding to 
input digit + action.

The network architecture:
1. Encoder: CNN to process MNIST image (28x28) -> feature vector
2. Action fusion: Concatenate image features with one-hot action vector
3. Decoder: MLP to generate output MNIST image (28x28)

Training data: For each sample, we take an MNIST image, apply a random action, and the target
is a random MNIST image from the class (original_digit + action) % 10.
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


class MNISTActionDataset(Dataset):
    """
    Custom dataset for MNIST with actions.
    
    For each sample:
    - Input: MNIST image + one-hot encoded action
    - Target: Random MNIST image from class (original_digit + action) % 10
    """
    
    def __init__(self, mnist_dataset, max_move=2, transform=None):
        self.mnist = mnist_dataset
        self.max_move = max_move
        self.action_space = 2 * max_move + 1  # [-max_move, ..., max_move]
        self.transform = transform
        
        # Build index for each digit class for fast sampling
        self.class_to_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.mnist):
            self.class_to_indices[label].append(idx)
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        # Get original image and label
        img, label = self.mnist[idx]
        
        # Sample random action in [-max_move, max_move]
        action = random.randint(-self.max_move, self.max_move)
        
        # One-hot encode action
        action_onehot = torch.zeros(self.action_space)
        action_onehot[action + self.max_move] = 1.0
        
        # Compute target class: (label + action) % 10
        target_class = (label + action) % 10
        
        # Sample a random image from the target class
        target_indices = self.class_to_indices[target_class]
        target_idx = random.choice(target_indices)
        target_img, _ = self.mnist[target_idx]
        
        # Apply transforms if provided
        if self.transform:
            img = self.transform(img)
            target_img = self.transform(target_img)
        
        return img, action_onehot, target_img, label, action, target_class


class MNISTActionTransformer(nn.Module):
    """
    Neural network that transforms MNIST digits based on actions.
    
    Architecture:
    1. Image Encoder: CNN to extract features from MNIST image
    2. Action Fusion: Concatenate image features with action vector
    3. Image Decoder: MLP to generate output MNIST image
    """
    
    def __init__(self, max_move=2, hidden_dim=256, latent_dim=128):
        super(MNISTActionTransformer, self).__init__()
        
        self.max_move = max_move
        self.action_space = 2 * max_move + 1
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Image Encoder: CNN to extract features from 28x28 MNIST image
        self.image_encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 7x7 -> 4x4 (using kernel_size=3, stride=2, padding=1)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Flatten: 128 * 4 * 4 = 2048
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Action fusion: Combine image features with action
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_space, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Image Decoder: Generate output MNIST image
        self.image_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 128 * 4 * 4),
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
    
    def forward(self, input_img, action):
        # Encode input image
        img_features = self.image_encoder(input_img)
        
        # Fuse image features with action
        fused_features = torch.cat([img_features, action], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # Decode to output image
        output_img = self.image_decoder(fused_features)
        
        return output_img


def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3, save_dir='./checkpoints'):
    """
    Train the MNIST Action Transformer model.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
            input_img, action, target_img, _, _, _ = batch
            input_img = input_img.to(device)
            action = action.to(device)
            target_img = target_img.to(device)
            
            optimizer.zero_grad()
            output_img = model(input_img, action)
            loss = criterion(output_img, target_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                input_img, action, target_img, _, _, _ = batch
                input_img = input_img.to(device)
                action = action.to(device)
                target_img = target_img.to(device)
                
                output_img = model(input_img, action)
                loss = criterion(output_img, target_img)
                
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
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device, max_move=2):
    """
    Evaluate the model and compute accuracy metrics.
    """
    model.eval()
    
    # Load a pre-trained MNIST classifier for evaluation
    try:
        from torchvision.models import resnet18
        classifier = resnet18(pretrained=False)
        classifier.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        classifier.fc = nn.Linear(classifier.fc.in_features, 10)
        
        # Try to load a pre-trained classifier, or train a simple one
        classifier_path = './mnist_classifier.pth'
        if os.path.exists(classifier_path):
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        else:
            print("Training a simple MNIST classifier for evaluation...")
            classifier = train_mnist_classifier()
            torch.save(classifier.state_dict(), classifier_path)
        
        classifier = classifier.to(device)
        classifier.eval()
        
    except Exception as e:
        print(f"Could not load classifier: {e}")
        classifier = None
    
    total_samples = 0
    correct_predictions = 0
    mse_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            input_img, action, target_img, _, _, target_labels = batch
            input_img = input_img.to(device)
            action = action.to(device)
            target_img = target_img.to(device)
            target_labels = target_labels.to(device)
            
            # Generate output
            output_img = model(input_img, action)
            
            # Compute MSE loss
            mse_loss += F.mse_loss(output_img, target_img).item()
            
            # Use classifier to check if output belongs to correct class
            if classifier is not None:
                # Resize output to match classifier input (224x224)
                output_resized = F.interpolate(output_img, size=(224, 224), mode='bilinear', align_corners=False)
                predictions = classifier(output_resized)
                predicted_labels = torch.argmax(predictions, dim=1)
                correct_predictions += (predicted_labels == target_labels).sum().item()
            
            total_samples += input_img.size(0)
    
    avg_mse = mse_loss / len(test_loader)
    accuracy = correct_predictions / total_samples if classifier is not None else 0.0
    
    print(f'Evaluation Results:')
    print(f'MSE Loss: {avg_mse:.4f}')
    if classifier is not None:
        print(f'Classification Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})')
    
    return avg_mse, accuracy


def train_mnist_classifier():
    """
    Train a simple MNIST classifier for evaluation purposes.
    """
    from torchvision.models import resnet18
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))  # Resize for ResNet
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    classifier = resnet18(pretrained=False)
    classifier.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    classifier.fc = nn.Linear(classifier.fc.in_features, 10)
    
    classifier = classifier.to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):  # Quick training
        classifier.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(classifier.conv1.weight.device), target.to(classifier.conv1.weight.device)
            optimizer.zero_grad()
            output = classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return classifier


def visualize_results(model, test_loader, device, max_move=2, num_samples=8):
    """
    Visualize model results by showing input, target, and generated images.
    """
    model.eval()
    
    # Get a batch of test data
    batch = next(iter(test_loader))
    input_imgs, actions, target_imgs, original_labels, action_values, target_labels = batch
    input_imgs = input_imgs[:num_samples].to(device)
    actions = actions[:num_samples].to(device)
    target_imgs = target_imgs[:num_samples]
    original_labels = original_labels[:num_samples]
    action_values = action_values[:num_samples]
    target_labels = target_labels[:num_samples]
    
    with torch.no_grad():
        output_imgs = model(input_imgs, actions)
    
    # Convert to numpy for plotting
    input_imgs = input_imgs.cpu().numpy()
    output_imgs = output_imgs.cpu().numpy()
    target_imgs = target_imgs.numpy()
    
    # Create visualization
    fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_samples):
        # Input image
        axes[0, i].imshow(input_imgs[i, 0], cmap='gray')
        axes[0, i].set_title(f'Input: {original_labels[i].item()}\nAction: {action_values[i].item()}')
        axes[0, i].axis('off')
        
        # Target image
        axes[1, i].imshow(target_imgs[i, 0], cmap='gray')
        axes[1, i].set_title(f'Target: {target_labels[i].item()}')
        axes[1, i].axis('off')
        
        # Generated image
        axes[2, i].imshow(output_imgs[i, 0], cmap='gray')
        axes[2, i].set_title(f'Generated: {target_labels[i].item()}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses, val_losses):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    """
    Main function to train and evaluate the MNIST Action Transformer.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Configuration
    max_move = 2  # Action range: [-2, -1, 0, 1, 2]
    batch_size = 128
    epochs = 50
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Using device: {device}')
    print(f'Max move: {max_move}, Action space size: {2*max_move+1}')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Load MNIST datasets
    train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create custom datasets
    train_dataset = MNISTActionDataset(train_mnist, max_move=max_move)
    test_dataset = MNISTActionDataset(test_mnist, max_move=max_move)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # Create model
    model = MNISTActionTransformer(max_move=max_move, hidden_dim=256, latent_dim=128)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Split training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Train model
    print('Starting training...')
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device, 
        epochs=epochs, lr=lr, save_dir='./checkpoints'
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Load best model
    checkpoint = torch.load('./checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    print('Evaluating model...')
    mse_loss, accuracy = evaluate_model(model, test_loader, device, max_move)
    
    # Visualize results
    print('Visualizing results...')
    visualize_results(model, test_loader, device, max_move, num_samples=8)
    
    print('Training and evaluation complete!')


if __name__ == '__main__':
    main()
