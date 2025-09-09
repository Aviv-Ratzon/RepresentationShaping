"""
Test script for the MNIST Action Transformer.
This script tests the basic functionality without full training.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from mnist_action_transformer import MNISTActionTransformer, MNISTActionDataset
import matplotlib.pyplot as plt
import numpy as np

def test_model_architecture():
    """Test that the model can be created and forward pass works."""
    print("Testing model architecture...")
    
    # Create model
    max_move = 2
    model = MNISTActionTransformer(max_move=max_move, hidden_dim=256, latent_dim=128)
    
    # Test forward pass
    batch_size = 4
    input_img = torch.randn(batch_size, 1, 28, 28)
    action = torch.zeros(batch_size, 2*max_move+1)
    action[:, max_move] = 1.0  # Action = 0
    
    output = model(input_img, action)
    
    print(f"Input shape: {input_img.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    assert output.shape == input_img.shape, f"Output shape {output.shape} doesn't match input shape {input_img.shape}"
    print("✓ Model architecture test passed!")


def test_dataset():
    """Test that the dataset can be created and samples can be loaded."""
    print("\nTesting dataset...")
    
    # Create a small MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Use a small subset for testing
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Create custom dataset
    max_move = 2
    custom_dataset = MNISTActionDataset(mnist_dataset, max_move=max_move)
    
    # Test getting a sample
    input_img, action, target_img, original_label, action_value, target_label = custom_dataset[0]
    
    print(f"Input image shape: {input_img.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Target image shape: {target_img.shape}")
    print(f"Original label: {original_label}")
    print(f"Action value: {action_value}")
    print(f"Target label: {target_label}")
    
    # Verify action encoding
    expected_action_idx = action_value + max_move
    assert action[expected_action_idx] == 1.0, "Action encoding incorrect"
    assert action.sum() == 1.0, "Action should be one-hot"
    
    # Verify target label calculation
    expected_target = (original_label + action_value) % 10
    assert target_label == expected_target, f"Target label {target_label} doesn't match expected {expected_target}"
    
    print("✓ Dataset test passed!")


def test_training_step():
    """Test a single training step."""
    print("\nTesting training step...")
    
    # Create model and data
    max_move = 2
    model = MNISTActionTransformer(max_move=max_move, hidden_dim=64, latent_dim=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create dummy data
    batch_size = 4
    input_img = torch.randn(batch_size, 1, 28, 28)
    action = torch.zeros(batch_size, 2*max_move+1)
    action[:, max_move+1] = 1.0  # Action = 1
    target_img = torch.randn(batch_size, 1, 28, 28)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    output = model(input_img, action)
    loss = criterion(output, target_img)
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.4f}")
    print("✓ Training step test passed!")


def visualize_sample():
    """Visualize a sample from the dataset."""
    print("\nCreating visualization...")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    custom_dataset = MNISTActionDataset(mnist_dataset, max_move=2)
    
    # Get a sample
    input_img, action, target_img, original_label, action_value, target_label = custom_dataset[0]
    
    # Denormalize for visualization
    def denormalize(tensor):
        return (tensor * 0.5) + 0.5
    
    input_img = denormalize(input_img).squeeze()
    target_img = denormalize(target_img).squeeze()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title(f'Input: Digit {original_label}\nAction: {action_value}')
    axes[0].axis('off')
    
    axes[1].imshow(target_img, cmap='gray')
    axes[1].set_title(f'Target: Digit {target_label}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization created and saved as 'sample_visualization.png'")


def main():
    """Run all tests."""
    print("Running MNIST Action Transformer Tests")
    print("=" * 50)
    
    try:
        test_model_architecture()
        test_dataset()
        test_training_step()
        visualize_sample()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! The implementation is working correctly.")
        print("\nTo run full training, execute:")
        print("python mnist_action_transformer.py")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
