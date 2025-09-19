"""
Standalone sequence prediction task implementation.

This module implements a simplified version of the representation shaping task where:
- Input: one-hot encoded starting state + series of one-hot encoded actions
- Target: resulting states after performing the action sequence
- Model: deep linear or nonlinear network trained with CrossEntropy loss

To switch between linear and nonlinear networks, change the 'nonlinear' parameter in main():
- nonlinear = False: Pure linear network (no activation functions)
- nonlinear = True: Nonlinear network with ReLU activation functions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm


def one_hot(x, num_classes):
    """Convert integer to one-hot encoding."""
    return np.eye(num_classes)[x]


def generate_sequence_data(S, A, num_samples=1000, seed=0):
    """
    Generate data for sequence prediction task.
    
    Args:
        S: Number of states
        A: Number of actions in sequence
        num_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        X: Input data of shape (num_samples, S + A*3) - starting state + action sequence
        y: Target data of shape (num_samples, S*A) - resulting states after each action
    """
    np.random.seed(seed)
    
    X = []
    y = []
    
    for _ in range(num_samples):
        # Generate random starting state
        start_state = np.random.randint(0, S)
        start_state_onehot = one_hot(start_state, S)
        
        # Generate random action sequence
        actions = np.random.randint(-1, 2, A)  # Actions from [-1, 0, 1]
        action_sequence = []
        for action in actions:
            action_onehot = one_hot(action + 1, 3)  # Convert [-1,0,1] to [0,1,2]
            action_sequence.extend(action_onehot)
        
        # Compute resulting states after each action
        current_state = start_state
        resulting_states = []
        
        for action in actions:
            # Apply action (simple addition with reflective boundary conditions)
            next_state = current_state + action
            # Apply reflective boundary conditions
            if next_state < 0:
                next_state = -next_state
            if next_state >= S:
                next_state = 2 * (S - 1) - next_state
            resulting_states.append(one_hot(next_state, S))
            current_state = next_state
        # Flatten resulting states
        resulting_states_flat = np.concatenate(resulting_states)
        
        # Combine starting state and action sequence
        input_vec = np.concatenate([start_state_onehot, action_sequence])
        
        X.append(input_vec)
        y.append(resulting_states_flat)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class DeepLinearNetwork(nn.Module):
    """Deep linear network for sequence prediction."""
    
    def __init__(self, input_size, output_size, num_layers=8, nonlinear=False):
        super(DeepLinearNetwork, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nonlinear
        
        # Hidden dimension is max of input and output dimensions
        hidden_size = max(input_size, output_size)
        
        # Create layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.layers = nn.ModuleList(layers)
        
        # Activation function
        if nonlinear:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using Xavier normal."""
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation function to all layers except the last
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


def train_model(model, X, y, num_epochs=1000, learning_rate=0.01, batch_size=32, S=10, A=3, nonlinear=False):
    """
    Train the model on sequence prediction task.
    
    Args:
        model: The neural network model
        X: Input data
        y: Target data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        S: Number of states
        A: Number of actions
        nonlinear: Whether to use nonlinear activation functions
        
    Returns:
        loss_history: List of training losses
        accuracy_history: List of training accuracies
        hidden_states: List of hidden states for PCA analysis
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_history = []
    accuracy_history = []
    hidden_states = []
    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Create batches
        indices = torch.randperm(X.size(0))
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for i in range(0, X.size(0), batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            
            # Reshape outputs and targets for CrossEntropy loss
            # Each output should be treated as separate classification problems
            batch_size_actual = outputs.size(0)
            outputs_reshaped = outputs.view(batch_size_actual * A, S)
            y_reshaped = y_batch.view(batch_size_actual * A, S)
            
            # Convert to class indices for CrossEntropy loss
            y_classes = torch.argmax(y_reshaped, dim=1)
            
            # Compute loss
            loss = criterion(outputs_reshaped, y_classes)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            with torch.no_grad():
                predicted_classes = torch.argmax(outputs_reshaped, dim=1)
                accuracy = (predicted_classes == y_classes).float().mean().item()
                total_accuracy += accuracy
            
            num_batches += 1
        
        # Store metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        loss_history.append(avg_loss)
        accuracy_history.append(avg_accuracy)
        
        # Store hidden states for PCA analysis (every 100 epochs)
        if epoch % 100 == 0:
            with torch.no_grad():
                # Get the last hidden layer activations
                x_temp = X
                for i, layer in enumerate(model.layers[:-1]):  # All layers except the last
                    x_temp = layer(x_temp)
                hidden_states.append(x_temp.cpu().numpy())
    
    return loss_history, accuracy_history, hidden_states


def plot_training_dynamics(loss_history, accuracy_history):
    """Plot training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(loss_history)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(accuracy_history)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_pca_activations(hidden_states, y, S, A, epoch_indices=None):
    """
    Plot PCA of last layer activations colored by final target states.
    
    Args:
        hidden_states: List of hidden state arrays from different epochs
        y: Target data
        S: Number of states
        A: Number of actions
        epoch_indices: Which epochs to plot (if None, plot all)
    """
    if epoch_indices is None:
        epoch_indices = range(len(hidden_states))
    
    num_plots = len(epoch_indices)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    # Get final target states (last A states in the sequence)
    final_states = y[:, -S:]  # Last S elements (final state)
    final_state_classes = np.argmax(final_states, axis=1)
    
    for i, epoch_idx in enumerate(epoch_indices):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Perform PCA
        pca = PCA(n_components=2)
        hidden_2d = pca.fit_transform(hidden_states[epoch_idx])
        
        # Plot with colors based on final target states
        scatter = ax.scatter(hidden_2d[:, 0], hidden_2d[:, 1], 
                           c=final_state_classes, cmap='coolwarm', alpha=0.6)
        ax.set_title(f'Epoch {epoch_idx * 100}')
        ax.axis('equal')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Final Target State')
    
    # Hide empty subplots
    for i in range(num_plots, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_pca_comparison(all_hidden_states, all_y, S, A_values):
    """
    Plot PCA of last layer activations for different A values in a single plot.
    
    Args:
        all_hidden_states: List of hidden state arrays for each A value
        all_y: List of target data for each A value
        S: Number of states
        A_values: List of A values tested
    """
    # Calculate subplot layout
    num_plots = len(A_values)
    cols = 5  # 5 columns
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    if num_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    for i, (A, hidden_states, y) in enumerate(zip(A_values, all_hidden_states, all_y)):
        # Get final target states (last S elements - final state)
        final_states = y[:, -S:]  # Last S elements (final state)
        final_state_classes = np.argmax(final_states, axis=1)
        
        # Perform PCA
        pca = PCA(n_components=2)
        hidden_2d = pca.fit_transform(hidden_states)
        
        # Plot with colors based on final target states
        scatter = axes[i].scatter(hidden_2d[:, 0], hidden_2d[:, 1], 
                                c=final_state_classes, cmap='tab10', alpha=0.6, s=20)
        axes[i].set_title(f'A = {A}')
        axes[i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the sequence prediction task for A = 1:20."""
    # Parameters
    S = 10  # Number of states
    A_values = list(range(1, 21))  # A = 1:20
    num_samples = 1000
    num_epochs = 1000
    learning_rate = 0.01
    batch_size = 32
    num_layers = 8
    nonlinear = True  # Set to True for nonlinear network
    
    print(f"Sequence Prediction Task - A = 1:20")
    print(f"States: {S}, Actions per sequence: {A_values}")
    print(f"Samples: {num_samples}, Epochs: {num_epochs}")
    print(f"Network type: {'Nonlinear' if nonlinear else 'Linear'}")
    print()
    
    all_hidden_states = []
    all_y = []
    
    # Train models for each A value
    for A in A_values:
        print(f"Training model for A = {A}...")
        
        # Generate data
        X, y = generate_sequence_data(S, A, num_samples, seed=42)
        
        # Create model
        input_size = S + A * 3
        output_size = S * A
        model = DeepLinearNetwork(input_size, output_size, num_layers, nonlinear=nonlinear)
        
        # Train model
        loss_history, accuracy_history, hidden_states = train_model(
            model, X, y, num_epochs, learning_rate, batch_size, S, A, nonlinear
        )
        
        # Store final hidden states and targets
        all_hidden_states.append(hidden_states[-1])  # Last epoch hidden states
        all_y.append(y)
        
        print(f"A = {A}: Final loss: {loss_history[-1]:.4f}, Final accuracy: {accuracy_history[-1]:.4f}")
    
    print("\nTraining completed for all A values!")
    print("Plotting PCA comparison...")
    
    # Plot PCA comparison
    plot_pca_comparison(all_hidden_states, all_y, S, A_values)


if __name__ == "__main__":
    main()
