"""
Standalone Predictive Coding Network (PCN) for Corridor Task

This script implements a Predictive Coding Network that learns to predict
the next state of an agent moving in a corridor, given the current state
and action. States and actions are represented using one-hot encoding.

Predictive Coding Networks use hierarchical prediction errors to learn
representations, where each layer predicts the layer below it.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools


class CorridorDataset(Dataset):
    """Dataset for corridor task with one-hot encoded states and actions."""
    
    def __init__(self, corridor_length=30, max_move=10, min_move=1, allow_backwards=True):
        """
        Initialize corridor dataset.
        
        Args:
            corridor_length: Length of the corridor (number of positions)
            max_move: Maximum number of steps the agent can move
            min_move: Minimum number of steps (usually 1)
            allow_backwards: Whether to allow negative movements
        """
        self.corridor_length = corridor_length
        self.max_move = max_move
        self.min_move = min_move
        self.allow_backwards = allow_backwards
        
        # Generate all possible actions
        if allow_backwards:
            actions = np.concatenate([
                np.arange(-max_move, -min_move + 1),
                np.arange(min_move, max_move + 1)
            ])
        else:
            actions = np.arange(min_move, max_move + 1)
        self.actions = np.unique(actions)
        self.n_actions = len(self.actions)
        
        # Generate all state-action-next_state triplets
        self.data = self._generate_data()
        
    def _one_hot(self, index, size):
        """Create one-hot encoding."""
        vec = np.zeros(size)
        vec[index] = 1.0
        return vec
    
    def _generate_data(self):
        """Generate all valid state-action-next_state transitions."""
        data = []
        
        for state_idx in range(self.corridor_length):
            state_one_hot = self._one_hot(state_idx, self.corridor_length)
            
            for action in self.actions:
                next_state_idx = state_idx + action
                
                # Check if next state is valid (within corridor bounds)
                if 0 <= next_state_idx < self.corridor_length:
                    action_one_hot = self._one_hot(
                        np.where(self.actions == action)[0][0],
                        self.n_actions
                    )
                    next_state_one_hot = self._one_hot(next_state_idx, self.corridor_length)
                    
                    # Input: concatenate state and action
                    input_vec = np.concatenate([state_one_hot, action_one_hot])
                    
                    data.append({
                        'input': input_vec,
                        'target': next_state_one_hot,
                        'state_idx': state_idx,
                        'action': action,
                        'next_state_idx': next_state_idx
                    })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input': torch.FloatTensor(item['input']),
            'target': torch.FloatTensor(item['target']),
            'state_idx': item['state_idx'],
            'action': item['action'],
            'next_state_idx': item['next_state_idx']
        }


class PredictiveCodingLayer(nn.Module):
    """A single layer in a Predictive Coding Network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize a PCN layer.
        
        Args:
            input_size: Size of input from layer below
            hidden_size: Size of hidden representation
            output_size: Size of output (prediction of layer below)
        """
        super(PredictiveCodingLayer, self).__init__()
        
        # Bottom-up pathway: input -> hidden representation
        self.bottom_up = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        # Top-down pathway: hidden -> prediction of layer below
        self.top_down = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Use sigmoid for one-hot like outputs
        )
        
    def forward(self, x, prediction_error=None):
        """
        Forward pass through PCN layer.
        
        Args:
            x: Input from layer below
            prediction_error: Optional prediction error to incorporate
        """
        # Compute hidden representation
        h = self.bottom_up(x)
        
        # Predict layer below
        prediction = self.top_down(h)
        
        # If prediction error is provided, incorporate it
        if prediction_error is not None:
            h = h + prediction_error
        
        return h, prediction


class PredictiveCodingNetwork(nn.Module):
    """Predictive Coding Network for corridor task."""
    
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64, 32]):
        """
        Initialize PCN.
        
        Args:
            input_size: Size of input (state + action)
            output_size: Size of output (next state)
            hidden_sizes: List of hidden layer sizes (from bottom to top)
        """
        super(PredictiveCodingNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes)
        
        # Build layers from bottom to top
        self.layers = nn.ModuleList()
        
        # First layer: input -> first hidden
        self.layers.append(
            PredictiveCodingLayer(input_size, hidden_sizes[0], input_size)
        )
        
        # Intermediate layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                PredictiveCodingLayer(
                    hidden_sizes[i-1],
                    hidden_sizes[i],
                    hidden_sizes[i-1]
                )
            )
        
        # Top layer: highest hidden -> output prediction
        self.top_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], output_size),
            nn.Softmax(dim=1)  # Softmax for one-hot like outputs
        )
        
    def forward(self, x, n_inference_steps=3, inference_rate=0.1):
        """
        Forward pass with iterative inference (simplified for training).
        
        Args:
            x: Input tensor [batch_size, input_size]
            n_inference_steps: Number of inference iterations (not used in simplified version)
            inference_rate: Learning rate for inference updates (not used in simplified version)
        """
        # Simplified forward pass: standard feedforward with PCN structure
        # In a full PCN, this would involve iterative inference, but for training
        # efficiency, we use a standard forward pass
        
        # Bottom-up pass
        current_input = x
        hidden_states = []
        
        for layer in self.layers:
            h, prediction = layer(current_input)
            hidden_states.append(h)
            current_input = h
        
        # Top-down: generate output prediction
        output = self.top_layer(hidden_states[-1])
        
        return output, hidden_states
    
    def compute_prediction_errors(self, x, target, hidden_states):
        """Compute prediction errors for all layers."""
        errors = []
        
        # Error at input level: how well does layer 0 predict the input?
        input_pred = self.layers[0].top_down(hidden_states[0])
        input_error = x - input_pred
        errors.append(input_error)
        
        # Errors at hidden levels: how well does each layer predict the layer below?
        for i in range(len(self.layers) - 1):
            pred = self.layers[i+1].top_down(hidden_states[i+1])
            error = hidden_states[i] - pred
            errors.append(error)
        
        # Error at output level: how well does top layer predict the target?
        output_pred = self.top_layer(hidden_states[-1])
        output_error = target - output_pred
        errors.append(output_error)
        
        return errors


def train_pcn(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001,
              device='cpu', print_every=10):
    """
    Train the Predictive Coding Network.
    
    Args:
        model: PCN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        print_every: Print metrics every N epochs
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Use CrossEntropyLoss for one-hot targets (treat as classification)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, hidden_states = model(inputs)
            
            # Compute loss
            # Convert one-hot targets to class indices for CrossEntropyLoss
            target_classes = targets.argmax(dim=1)
            loss = criterion(outputs, target_classes)
            
            # Add prediction error terms (PCN-specific)
            prediction_errors = model.compute_prediction_errors(inputs, targets, hidden_states)
            error_loss = sum([torch.mean(err**2) for err in prediction_errors])
            
            total_loss = loss + 0.1 * error_loss  # Weight prediction errors
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Compute accuracy
            predicted_classes = outputs.argmax(dim=1)
            train_correct += (predicted_classes == target_classes).sum().item()
            train_total += target_classes.size(0)
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                outputs, _ = model(inputs)
                target_classes = targets.argmax(dim=1)
                loss = criterion(outputs, target_classes)
                
                predicted_classes = outputs.argmax(dim=1)
                val_correct += (predicted_classes == target_classes).sum().item()
                val_total += target_classes.size(0)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
        
        if (epoch + 1) % print_every == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            print()
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }


def plot_training_curves(history):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history['train_losses'], label='Train Loss')
    axes[0].plot(history['val_losses'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy curves
    axes[1].plot(history['train_accuracies'], label='Train Accuracy')
    axes[1].plot(history['val_accuracies'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('pcn_training_curves.png', dpi=150)
    print("Training curves saved to 'pcn_training_curves.png'")
    plt.show()


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the trained model on test data."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs, _ = model(inputs)
            predicted_classes = outputs.argmax(dim=1)
            target_classes = targets.argmax(dim=1)
            
            correct += (predicted_classes == target_classes).sum().item()
            total += target_classes.size(0)
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy


def main():
    """Main function to run PCN training on corridor task."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = {
        'corridor_length': 30,
        'max_move': 10,
        'min_move': 1,
        'allow_backwards': True,
        'hidden_sizes': [128, 64, 32],
        'num_epochs': 1000,
        'learning_rate': 0.001,
        'batch_size': 32,
        'train_split': 0.7,
        'val_split': 0.15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 60)
    print("Predictive Coding Network - Corridor Task")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create dataset
    print("Generating corridor dataset...")
    dataset = CorridorDataset(
        corridor_length=config['corridor_length'],
        max_move=config['max_move'],
        min_move=config['min_move'],
        allow_backwards=config['allow_backwards']
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Input size: {dataset[0]['input'].size(0)}")
    print(f"Output size: {dataset[0]['target'].size(0)}")
    print()
    
    # Split dataset
    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    # Create model
    input_size = dataset[0]['input'].size(0)
    output_size = dataset[0]['target'].size(0)
    
    print("Creating Predictive Coding Network...")
    model = PredictiveCodingNetwork(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=config['hidden_sizes']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train model
    print("Starting training...")
    history = train_pcn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device'],
        print_every=10
    )
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {history['best_val_acc']:.4f}")
    print()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader, device=config['device'])
    
    # Plot training curves
    plot_training_curves(history)
    
    # Example prediction
    print("\nExample predictions:")
    model.eval()
    with torch.no_grad():
        sample = dataset[0]
        input_tensor = sample['input'].unsqueeze(0).to(config['device'])
        output, _ = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()
        target_idx = sample['target'].argmax().item()
        
        print(f"State: {sample['state_idx']}, Action: {sample['action']}")
        print(f"Predicted next state: {predicted_idx}, Actual: {target_idx}")
        print(f"Correct: {predicted_idx == target_idx}")


if __name__ == '__main__':
    main()

