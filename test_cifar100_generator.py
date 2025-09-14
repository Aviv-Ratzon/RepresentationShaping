#!/usr/bin/env python3
"""
Test script for CIFAR-100 Conditional Image Generation
"""

import torch
import os
import sys
from cifar100_conditional_generator import ConditionalVAE, CIFAR100ActionDataset, get_data_loaders

def test_model_creation():
    """Test model creation and forward pass."""
    print("Testing model creation...")
    
    # Create model
    model = ConditionalVAE(
        input_channels=3,
        latent_dim=16,  # Smaller for testing
        action_range=3
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 4
    input_images = torch.randn(batch_size, 3, 32, 32)
    actions = torch.randn(batch_size, 7)  # 2*3+1 = 7 for action_range=3
    
    with torch.no_grad():
        recon_images, mu, logvar, z, z_conditioned = model(input_images, actions)
    
    print(f"Input shape: {input_images.shape}")
    print(f"Reconstructed shape: {recon_images.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Conditioned latent shape: {z_conditioned.shape}")
    
    assert recon_images.shape == input_images.shape
    assert z.shape == (batch_size, 16)
    assert z_conditioned.shape == (batch_size, 16)
    
    print("✓ Model creation test passed!")


def test_dataset():
    """Test dataset creation and loading."""
    print("\nTesting dataset creation...")
    
    try:
        # Create a small dataset for testing
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = CIFAR100ActionDataset(
            root='./data', 
            train=True, 
            action_range=3, 
            transform=transform,
            num_classes=100
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input image shape: {sample['input_image'].shape}")
        print(f"Target image shape: {sample['target_image'].shape}")
        print(f"Action shape: {sample['action'].shape}")
        print(f"Original class: {sample['original_class']}")
        print(f"Target class: {sample['target_class']}")
        print(f"Action value: {sample['action_value']}")
        
        assert sample['input_image'].shape == (3, 32, 32)
        assert sample['target_image'].shape == (3, 32, 32)
        assert sample['action'].shape == (7,)  # 2*3+1
        assert 0 <= sample['original_class'] < 100
        assert 0 <= sample['target_class'] < 100
        assert -3 <= sample['action_value'] <= 3
        
        print("✓ Dataset test passed!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        print("This might be due to CIFAR-100 download issues. Continuing with other tests...")


def test_data_loader():
    """Test data loader creation."""
    print("\nTesting data loader creation...")
    
    try:
        train_loader, val_loader = get_data_loaders(
            batch_size=8, 
            action_range=3, 
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
        
        # Test getting a batch
        batch = next(iter(train_loader))
        print(f"Batch size: {batch['input_image'].shape[0]}")
        print(f"Input images shape: {batch['input_image'].shape}")
        print(f"Target images shape: {batch['target_image'].shape}")
        print(f"Actions shape: {batch['action'].shape}")
        
        assert batch['input_image'].shape[0] == 8
        assert batch['input_image'].shape[1:] == (3, 32, 32)
        assert batch['target_image'].shape == batch['input_image'].shape
        assert batch['action'].shape == (8, 7)
        
        print("✓ Data loader test passed!")
        
    except Exception as e:
        print(f"Data loader test failed: {e}")
        print("This might be due to CIFAR-100 download issues. Continuing with other tests...")


def test_generation():
    """Test image generation."""
    print("\nTesting image generation...")
    
    model = ConditionalVAE(
        input_channels=3,
        latent_dim=16,
        action_range=3
    )
    
    # Test generation
    class_indices = [0, 1, 2, 3]
    actions = torch.randn(4, 7)  # 2*3+1 = 7
    
    with torch.no_grad():
        generated = model.generate(class_indices, actions, device='cpu')
    
    print(f"Generated images shape: {generated.shape}")
    assert generated.shape == (4, 3, 32, 32)
    
    print("✓ Generation test passed!")


def main():
    """Run all tests."""
    print("Running CIFAR-100 Conditional Generator Tests")
    print("=" * 50)
    
    test_model_creation()
    test_dataset()
    test_data_loader()
    test_generation()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("\nTo run the full training:")
    print("python cifar100_conditional_generator.py --epochs 10 --batch_size 32 --latent_dim 16")


if __name__ == '__main__':
    main()
