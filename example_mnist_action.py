"""
Example usage of the MNIST Action Generator.

This script demonstrates how to train and use the MNIST action generator
for different configurations.
"""

import torch
import matplotlib.pyplot as plt
from mnist_action_generator import (
    MNISTActionDataset, 
    MNISTActionGenerator, 
    train_generator, 
    evaluate_model,
    visualize_results,
    train_classifier
)
from torch.utils.data import DataLoader
from torchvision import transforms


def example_cyclic_training():
    """
    Example: Train with cyclic arithmetic (label + action) % 10
    """
    print("=" * 50)
    print("CYCLIC TRAINING EXAMPLE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration for cyclic training
    config = {
        'max_action': 3,  # Actions in [-3, -2, -1, 0, 1, 2, 3]
        'cyclic': True,   # Use modular arithmetic
        'batch_size': 64,
        'epochs': 20,
        'lr': 1e-3
    }
    
    # Create datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = MNISTActionGenerator(
        max_action=config['max_action'],
        latent_dim=128,
        hidden_dim=256
    )
    
    # Train model
    train_losses, val_losses = train_generator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config['epochs'],
        lr=config['lr'],
        save_dir='./checkpoints_cyclic',
        use_adversarial=False,
        lambda_clf=0.1
    )
    
    # Visualize results
    visualize_results(model, val_loader, device, config['max_action'])
    
    return model, train_losses, val_losses


def example_non_cyclic_training():
    """
    Example: Train with non-cyclic arithmetic (discard invalid labels)
    """
    print("=" * 50)
    print("NON-CYCLIC TRAINING EXAMPLE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration for non-cyclic training
    config = {
        'max_action': 2,  # Actions in [-2, -1, 0, 1, 2]
        'cyclic': False,  # Discard invalid labels
        'batch_size': 64,
        'epochs': 20,
        'lr': 1e-3
    }
    
    # Create datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = MNISTActionGenerator(
        max_action=config['max_action'],
        latent_dim=128,
        hidden_dim=256
    )
    
    # Train model
    train_losses, val_losses = train_generator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config['epochs'],
        lr=config['lr'],
        save_dir='./checkpoints_non_cyclic',
        use_adversarial=False,
        lambda_clf=0.1
    )
    
    # Visualize results
    visualize_results(model, val_loader, device, config['max_action'])
    
    return model, train_losses, val_losses


def example_adversarial_training():
    """
    Example: Train with adversarial loss for better quality
    """
    print("=" * 50)
    print("ADVERSARIAL TRAINING EXAMPLE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration for adversarial training
    config = {
        'max_action': 2,
        'cyclic': True,
        'batch_size': 64,
        'epochs': 30,
        'lr': 1e-3
    }
    
    # Create datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = MNISTActionGenerator(
        max_action=config['max_action'],
        latent_dim=128,
        hidden_dim=256
    )
    
    # Train model with adversarial loss
    train_losses, val_losses = train_generator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config['epochs'],
        lr=config['lr'],
        save_dir='./checkpoints_adversarial',
        use_adversarial=True,  # Enable adversarial training
        lambda_clf=0.1,
        lambda_adv=0.01
    )
    
    # Visualize results
    visualize_results(model, val_loader, device, config['max_action'])
    
    return model, train_losses, val_losses


def compare_configurations():
    """
    Compare different configurations and plot results
    """
    print("=" * 50)
    print("CONFIGURATION COMPARISON")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different max_action values
    max_actions = [1, 2, 3]
    results = {}
    
    for max_action in max_actions:
        print(f"\nTesting max_action = {max_action}")
        
        # Create datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = MNISTActionDataset(
            max_action=max_action,
            transform=transform,
            cyclic=True,
            train=True
        )
        
        val_dataset = MNISTActionDataset(
            max_action=max_action,
            transform=transform,
            cyclic=True,
            train=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Create and train model
        model = MNISTActionGenerator(
            max_action=max_action,
            latent_dim=128,
            hidden_dim=256
        )
        
        train_losses, val_losses = train_generator(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=15,
            lr=1e-3,
            save_dir=f'./checkpoints_compare_{max_action}',
            use_adversarial=False,
            lambda_clf=0.1
        )
        
        results[max_action] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': val_losses[-1]
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for max_action, result in results.items():
        plt.plot(result['val_losses'], label=f'max_action={max_action}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    max_actions_list = list(results.keys())
    final_losses = [results[ma]['final_val_loss'] for ma in max_actions_list]
    plt.bar(max_actions_list, final_losses)
    plt.xlabel('Max Action')
    plt.ylabel('Final Validation Loss')
    plt.title('Final Performance by Max Action')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    # Run examples
    print("MNIST Action Generator Examples")
    print("Choose an example to run:")
    print("1. Cyclic training")
    print("2. Non-cyclic training") 
    print("3. Adversarial training")
    print("4. Compare configurations")
    print("5. Run all examples")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        example_cyclic_training()
    elif choice == "2":
        example_non_cyclic_training()
    elif choice == "3":
        example_adversarial_training()
    elif choice == "4":
        compare_configurations()
    elif choice == "5":
        example_cyclic_training()
        example_non_cyclic_training()
        example_adversarial_training()
        compare_configurations()
    else:
        print("Invalid choice. Running cyclic training example...")
        example_cyclic_training()
