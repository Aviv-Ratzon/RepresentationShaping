import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Parameters
ds = 0.1
S = 1.0  # Range for s: [-S, S]
ds_plot = ds / 100  # Interval for plotting
v = np.random.randn(10)[None,:]

# Values of A to test
A_values = [ds/2, S/2]

# Define the neural network
class DeepFCN(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[100, 100, 100, 100], output_dim=1):
        super(DeepFCN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # self.rescale_parameters()

    def rescale_parameters(self):
        """
        Rescale parameters: multiply all parameters by 0.1,
        except for the 'fc' parameters, which are multiplied by 2.
        """
        # Multiply all parameters by 0.1
        for name, param in self.named_parameters():
            # Check if parameter belongs to fc layer
            if name.startswith("out"):
                param.data.mul_(10)
            else:
                param.data.mul_(0.1)
    
    def forward(self, x):
        h = self.network(x)
        out = self.output_layer(h)
        return out, h

def f(s, a=None):
    if a is None:
        a = np.zeros_like(s)
    return np.tanh(2 * np.pi * np.sin(2*np.pi*(s + a)[:,None]@v))


def generate_training_data(A, S, ds):
    """Generate training data for the given A value."""
    # Generate s values: [-S, S] with interval ds
    s_values = np.arange(-S, S + ds, ds)
    
    # Generate action values: [-A, A) with interval ds
    a_values = np.concatenate([np.arange(-ds, -A, -ds), np.arange(0, A, ds)])
    a_values = np.unique(a_values)
    
    # Create all combinations of s and a
    s_grid, a_grid = np.meshgrid(s_values, a_values)
    s_flat = s_grid.flatten()
    a_flat = a_grid.flatten()
    
    remove_idx = ((s_flat + a_flat) > S) | ((s_flat + a_flat) < -S)
    s_flat = s_flat[~remove_idx]
    a_flat = a_flat[~remove_idx]

    # Compute function samples: sin(2π * s)
    function_samples = f(s_flat)
    
    # Compute targets: sin(s + a)
    targets = f(s_flat, a_flat)
    
    # Input features: [sin(2π * s), a]
    inputs = np.column_stack([function_samples, a_flat])
    
    return inputs, targets, s_flat, a_flat

def train_model(A, S, ds, num_epochs=10, batch_size=256, learning_rate=0.001):
    """Train a deep fully connected network."""
    # Generate training data
    inputs, targets, s_vals, a_vals = generate_training_data(A, S, ds)
    
    # Convert to tensors
    X = torch.FloatTensor(inputs)
    y = torch.FloatTensor(targets)
    
    # Create model
    model = DeepFCN(input_dim=X.shape[1], hidden_dims=[128, 128, 128, 128], output_dim=y.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Track training loss
    train_losses = []
    
    # Training loop
    model.train()
    for epoch in tqdm(range(num_epochs), desc=f"Training (A={A})"):
        # Shuffle data
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Forward pass
            outputs, hidden = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Store average loss for this epoch
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
    
    return model, s_vals, a_vals, targets, train_losses

def plot_all_results(all_results, S, ds_plot):
    """Plot results from all A values in a single figure."""
    num_A = len(all_results)
    
    # Create figure with subplots: N rows (one per A), 4 columns (predictions, loss, PCA hidden, PCA targets)
    fig, axes = plt.subplots(num_A, 4, figsize=(3*4, 3*num_A))
    
    # Handle case where there's only one A value
    if num_A == 1:
        axes = axes.reshape(1, -1)
    
    # Generate plot data once (same for all A values)
    inputs_plot, targets_plot, s_vals_plot, a_vals_plot = generate_training_data(1e-10, 2*S, ds/100)
    
    # Compute color values: s_vals_plot + a_vals_plot
    color_values = s_vals_plot + a_vals_plot
    
    for idx, (A, model, s_train, a_train, targets_train, train_losses) in enumerate(all_results):
        # Get predictions from model
        model.eval()
        with torch.no_grad():
            X_plot = torch.FloatTensor(inputs_plot)
            predictions, hidden = model(X_plot)
            predictions = predictions.numpy()
            hidden = hidden.numpy()
        
        # Filter training samples where a=0 (for plotting)
        mask_a_zero = np.abs(a_train) < 1e-10  # Find samples where a ≈ 0
        s_train_a0 = s_train[mask_a_zero]
        targets_train_a0 = targets_train[mask_a_zero, 0]
        
        # Left subplot: Function predictions
        ax_pred = axes[idx, 0]
        ax_pred.plot(s_vals_plot, targets_plot[:,0], 'b-', label='True: f(x, 0)', linewidth=2)
        ax_pred.plot(s_vals_plot, predictions[:,0], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        ax_pred.scatter(s_train_a0, targets_train_a0, c='green', s=30, alpha=0.6, 
                        label='Training samples (a=0)', zorder=5)
        ax_pred.set_xlabel('x', fontsize=12)
        ax_pred.set_ylabel('Output', fontsize=12)
        ax_pred.set_title(f'Model Predictions vs True Values (A={A:.3f})', fontsize=14)
        ax_pred.legend(fontsize=11)
        ax_pred.grid(True, alpha=0.3)
        
        # Second subplot: Training loss curve
        ax_loss = axes[idx, 1]
        epochs = np.arange(1, len(train_losses) + 1)
        ax_loss.plot(epochs, train_losses, 'm-', linewidth=2, marker='o', markersize=4)
        ax_loss.set_xlabel('Epoch', fontsize=12)
        ax_loss.set_ylabel('Training Loss (MSE)', fontsize=12)
        ax_loss.set_title(f'Training Loss Curve (A={A:.3f})', fontsize=14)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale('log')  # Use log scale for better visualization
        
        # Third subplot: PCA of hidden activities
        ax_pca_hidden = axes[idx, 2]
        pca_hidden = PCA(n_components=2)
        hidden_pca = pca_hidden.fit_transform(hidden)
        scatter_hidden = ax_pca_hidden.scatter(hidden_pca[:, 0], hidden_pca[:, 1], 
                                                c=color_values, cmap='viridis', 
                                                s=10, alpha=0.6)
        ax_pca_hidden.set_xlabel(f'PC1 ({pca_hidden.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        ax_pca_hidden.set_ylabel(f'PC2 ({pca_hidden.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        ax_pca_hidden.set_title(f'PCA of Hidden Activities (A={A:.3f})', fontsize=14)
        plt.colorbar(scatter_hidden, ax=ax_pca_hidden, label='s+a')
        ax_pca_hidden.grid(True, alpha=0.3)
        
        # Fourth subplot: PCA of targets
        ax_pca_targets = axes[idx, 3]
        pca_targets = PCA(n_components=2)
        targets_pca = pca_targets.fit_transform(targets_plot)
        scatter_targets = ax_pca_targets.scatter(targets_pca[:, 0], targets_pca[:, 1], 
                                                  c=color_values, cmap='viridis', 
                                                  s=10, alpha=0.6)
        ax_pca_targets.set_xlabel(f'PC1 ({pca_targets.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        ax_pca_targets.set_ylabel(f'PC2 ({pca_targets.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        ax_pca_targets.set_title(f'PCA of Targets (A={A:.3f})', fontsize=14)
        plt.colorbar(scatter_targets, ax=ax_pca_targets, label='s+a')
        ax_pca_targets.grid(True, alpha=0.3)
        
        # Print error statistics
        mse = np.mean((predictions - targets_plot)**2)
        mae = np.mean(np.abs(predictions - targets_plot))
        print(f"\nA={A:.3f} - MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    plt.tight_layout()
    plt.savefig('plot_all_A_values.png', dpi=150)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Collect all results first
    all_results = []
    
    for A in A_values:
        print(f"\n{'='*60}")
        print(f"Processing A = {A}")
        print(f"{'='*60}")
        
        # Train model
        model, s_train, a_train, targets_train, train_losses = train_model(A, S, ds, num_epochs=100000, learning_rate=0.01)
        
        # Store results
        all_results.append((A, model, s_train, a_train, targets_train, train_losses))
        
        print(f"Completed processing for A = {A}\n")
    
    # Plot all results in a single figure
    plot_all_results(all_results, S, ds_plot)

