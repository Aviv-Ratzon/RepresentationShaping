import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Parameters
ds = 0.05
S = 1.0  # Range for s: [-S, S]
ds_plot = ds / 100  # Interval for plotting
v = np.random.randn(10)[None,:]

# Values of A to test
A_values = [ds, S/2]
lr = 0.00001
n_epochs = 100000
optimizer_fn = torch.optim.Adam

def poly(x):
    return -5*x**3 + 4*x**2 - 3*x + 2

def f(s, a=None):
    if a is None:
        a = np.zeros_like(s)        
    x = (s + a)[:,None]@v
    x_periodic = np.sin(2 * 2*np.pi*(s + a)[:,None]@v)
    return np.tanh(x_periodic)
    # return 0.1*poly(x_periodic)

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
    
    def forward(self, x):
        h = self.network(x)
        out = self.output_layer(h)
        return out, h

# Define a single hidden layer model
class SingleHiddenLayerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleHiddenLayerModel, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        h = self.activation(self.hidden(x))
        out = self.output_layer(h)
        return out


def generate_training_data(A, S, ds):
    """Generate training data for the given A value."""
    # Generate s values: [-S, S] with interval ds
    s_values = np.arange(-S, S + ds, ds)
    
    # Generate action values: [-A, A) with interval ds
    a_values = np.concatenate([np.arange(-ds, -A*1.0001, -ds), np.arange(0, A*1.0001, ds)])
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
    
    # Convert to tensors and move to device
    X = torch.FloatTensor(inputs).to(device)
    y = torch.FloatTensor(targets).to(device)
    
    # Create model and move to device
    model = DeepFCN(input_dim=X.shape[1], hidden_dims=[128, 128, 128, 128, 128, 128, 128, 128], output_dim=y.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
    
    # Track training loss
    train_losses = []
    
    # Training loop
    model.train()
    for epoch in tqdm(range(num_epochs), desc=f"Training (A={A})"):
        # Shuffle data
        indices = torch.randperm(len(X), device=device)
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

def train_hidden_to_target_model(original_model, X, y, hidden_dim=128, num_epochs=1000, batch_size=256, learning_rate=0.001):
    """Train a single hidden layer model to map hidden activations to targets."""
    # Get hidden activations from the original model
    inputs, targets, s_vals, a_vals = generate_training_data(1e-10, S, ds/100)
    
    # Convert to tensors and move to device
    X = torch.FloatTensor(inputs).to(device)
    y = torch.FloatTensor(targets).to(device)

    original_model.eval()
    with torch.no_grad():
        _, hidden_activations = original_model(X)
    
    # Get the hidden dimension size
    hidden_size = hidden_activations.shape[1]
    output_dim = y.shape[1]
    
    # Create the new model
    hidden_model = SingleHiddenLayerModel(input_dim=hidden_size, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(hidden_model.parameters(), lr=learning_rate)
    
    # Training loop
    hidden_model.train()
    for epoch in tqdm(range(num_epochs), desc="Training hidden-to-target model"):
        # Shuffle data
        indices = torch.randperm(len(hidden_activations), device=device)
        h_shuffled = hidden_activations[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(hidden_activations), batch_size):
            batch_h = h_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Forward pass
            outputs = hidden_model(batch_h)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Store average loss for this epoch
        avg_loss = epoch_loss / num_batches
    
    # Calculate final MSE
    hidden_model.eval()
    with torch.no_grad():
        final_outputs = hidden_model(hidden_activations)
        final_mse = criterion(final_outputs, y).item()
    
    print(f"Hidden-to-target model final MSE: {final_mse:.6f}")
    
    return hidden_model, final_mse

def plot_all_results(all_results, S, ds_plot):
    """Plot results from all A values in a single figure."""
    num_A = len(all_results)
    
    # Create figure with subplots: N rows (one per A), 4 columns (predictions, loss, PCA hidden, PCA targets)
    n_plots = 5
    fig, axes = plt.subplots(num_A, n_plots, figsize=(3*n_plots, 3*num_A))
    
    # Handle case where there's only one A value
    if num_A == 1:
        axes = axes.reshape(1, -1)
    
    # Generate plot data once (same for all A values)
    inputs_plot, targets_plot, s_vals_plot, a_vals_plot = generate_training_data(1e-10, S, ds/100)
    
    # Compute color values: s_vals_plot + a_vals_plot
    color_values = s_vals_plot + a_vals_plot
    
    for idx, (A, model, s_train, a_train, targets_train, train_losses) in enumerate(all_results):
        # Get predictions from model
        model.eval()
        with torch.no_grad():
            X_plot = torch.FloatTensor(inputs_plot).to(device)
            predictions, hidden = model(X_plot)
            predictions = predictions.cpu().numpy()
            hidden = hidden.cpu().numpy()
        
        # Calculate MSE
        mse = np.mean((predictions - targets_plot)**2)
        
        # Filter training samples where a=0 (for plotting)
        mask_a_zero = np.abs(a_train) < 1e-10  # Find samples where a ≈ 0
        s_train_a0 = s_train[mask_a_zero]
        targets_train_a0 = targets_train[mask_a_zero, 0]
        
        # Left subplot: Function predictions
        ax_i = 0
        ax_pred = axes[idx, ax_i]
        ax_pred.plot(s_vals_plot, targets_plot[:,0], 'b-', label='True: f(x, 0)', linewidth=2)
        ax_pred.plot(s_vals_plot, predictions[:,0], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        ax_pred.scatter(s_train_a0, targets_train_a0, c='green', s=30, alpha=0.6, 
                        label='Training samples (a=0)', zorder=5)
        ax_pred.set_xlabel('x', fontsize=6)
        ax_pred.set_ylabel('Output', fontsize=6)
        ax_pred.set_title(f'A={A:.3f}, MSE={mse:.6f}', fontsize=7)
        ax_pred.legend(fontsize=5.5)
        ax_pred.grid(True, alpha=0.3)
        
        # Left subplot: Function predictions
        targets_train_a0 = targets_train[mask_a_zero, 1]
        ax_i += 1
        ax_pred = axes[idx, ax_i]
        ax_pred.plot(s_vals_plot, targets_plot[:,1], 'b-', label='True: f(x, 0)', linewidth=2)
        ax_pred.plot(s_vals_plot, predictions[:,1], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        ax_pred.scatter(s_train_a0, targets_train_a0, c='green', s=30, alpha=0.6, 
                        label='Training samples (a=0)', zorder=5)
        ax_pred.set_xlabel('x', fontsize=6)
        ax_pred.set_ylabel('Output', fontsize=6)
        ax_pred.set_title(f'Model Predictions vs True Values (A={A:.3f})', fontsize=7)
        ax_pred.legend(fontsize=5.5)
        ax_pred.grid(True, alpha=0.3)
        
        # Second subplot: Training loss curve
        ax_i += 1
        ax_loss = axes[idx, ax_i]
        epochs = np.arange(1, len(train_losses) + 1)
        ax_loss.plot(epochs, train_losses, 'm-', linewidth=2, marker='o', markersize=4)
        ax_loss.set_xlabel('Epoch', fontsize=6)
        ax_loss.set_ylabel('Training Loss (MSE)', fontsize=6)
        ax_loss.set_title(f'Training Loss Curve (A={A:.3f})', fontsize=7)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale('log')  # Use log scale for better visualization
        
        # Third subplot: PCA of hidden activities
        ax_i += 1
        ax_pca_hidden = axes[idx, ax_i]
        pca_hidden = PCA(n_components=2)
        hidden_pca = pca_hidden.fit_transform(hidden)
        # Calculate R² between PCs and s+a
        reg_hidden = LinearRegression()
        reg_hidden.fit(hidden_pca[:,[0]], color_values)
        r2_hidden = r2_score(color_values, reg_hidden.predict(hidden_pca[:,[0]]))
        scatter_hidden = ax_pca_hidden.scatter(hidden_pca[:, 0], hidden_pca[:, 1], 
                                                c=color_values, cmap='viridis', 
                                                s=10, alpha=0.6)
        ax_pca_hidden.set_xlabel(f'PC1 ({pca_hidden.explained_variance_ratio_[0]:.2%} variance)', fontsize=6)
        ax_pca_hidden.set_ylabel(f'PC2 ({pca_hidden.explained_variance_ratio_[1]:.2%} variance)', fontsize=6)
        ax_pca_hidden.set_title(f'R²={r2_hidden:.3f}', fontsize=7)
        plt.colorbar(scatter_hidden, ax=ax_pca_hidden, label='s+a')
        ax_pca_hidden.grid(True, alpha=0.3)
        
        # Fourth subplot: PCA of targets
        ax_i += 1
        ax_pca_targets = axes[idx, ax_i]
        pca_targets = PCA(n_components=2)
        targets_pca = pca_targets.fit_transform(targets_plot)
        # Calculate R² between PCs and s+a
        reg_targets = LinearRegression()
        reg_targets.fit(targets_pca, color_values)
        r2_targets = r2_score(color_values, reg_targets.predict(targets_pca))
        scatter_targets = ax_pca_targets.scatter(targets_pca[:, 0], targets_pca[:, 1], 
                                                  c=color_values, cmap='viridis', 
                                                  s=10, alpha=0.6)
        ax_pca_targets.set_xlabel(f'PC1 ({pca_targets.explained_variance_ratio_[0]:.2%} variance)', fontsize=6)
        ax_pca_targets.set_ylabel(f'PC2 ({pca_targets.explained_variance_ratio_[1]:.2%} variance)', fontsize=6)
        ax_pca_targets.set_title(f'PCA of Targets (A={A:.3f}, R²={r2_targets:.3f})', fontsize=7)
        plt.colorbar(scatter_targets, ax=ax_pca_targets, label='s+a')
        ax_pca_targets.grid(True, alpha=0.3)
        
        # Print error statistics
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
        model, s_train, a_train, targets_train, train_losses = train_model(A, S, ds, num_epochs=n_epochs, learning_rate=lr)
        
        # Prepare data for hidden-to-target model training
        inputs, targets, s_vals, a_vals = generate_training_data(A, S, ds)
        X = torch.FloatTensor(inputs).to(device)
        y = torch.FloatTensor(targets).to(device)
        
        # Train hidden-to-target model
        hidden_model, hidden_mse = train_hidden_to_target_model(
            model, X, y, 
            hidden_dim=128, 
            num_epochs=1000, 
            learning_rate=0.001
        )
        
        # Store results
        all_results.append((A, model, s_train, a_train, targets_train, train_losses))
        
        print(f"Completed processing for A = {A}\n")
    
    # Plot all results in a single figure
    plot_all_results(all_results, S, ds_plot)

