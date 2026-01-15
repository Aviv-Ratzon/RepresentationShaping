import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from numpy.polynomial.chebyshev import chebvander
from itertools import product

import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)


def FW_transform(x, y):
    n=x.shape[1]
    f_hat = np.zeros(2**n)
    basis_bits_l = [
    [i for i, bit in enumerate(mask) if bit]
    for mask in product([0, 1], repeat=n)
    ]

    for i, S in enumerate(basis_bits_l):
        xi_i = np.prod(x[:, S], axis=1)
        f_hat[i] = np.mean(xi_i*y)
    return f_hat

def calc_degree(f_hat):
    n = int(np.log2(len(f_hat)))
    basis_bits_l = [
    [i for i, bit in enumerate(mask) if bit]
    for mask in product([0, 1], repeat=n)
    ]
    non_zero = f_hat!=0
    if np.sum(non_zero) == 0:
        return 0
    degree = max([len(basis_bits_l[i]) for i in np.where(non_zero)[0]])
    return degree

def array_to_binary(x: Iterable[Any]) -> Tuple[np.ndarray, Dict[Any, str]]:
    """
    Convert a 1D array-like x into a 2D binary code matrix using the minimal
    number of bits needed to represent all unique values.

    Steps:
      1) Find unique values in x (preserving first-appearance order)
      2) Assign each unique value an integer id 0..k-1
      3) Use b = ceil(log2(k)) bits (minimal), with a special case k<=1 -> b=1
      4) Return:
           - codes: shape (len(x), b) with 0/1 ints
           - mapping: dict {value: bitstring}

    Example:
      x = ["cat","dog","cat","bird"]
      k=3 -> b=2
      mapping might be {"cat":"00","dog":"01","bird":"10"}
      codes -> [[0,0],[0,1],[0,0],[1,0]]
    """
    x_list = list(x)
    n = len(x_list)

    # Unique values in order of first appearance
    mapping_id: Dict[Any, int] = {}
    uniques: List[Any] = []
    for v in x_list:
        if v not in mapping_id:
            mapping_id[v] = len(uniques)
            uniques.append(v)

    k = len(uniques)
    bits = 1 if k <= 1 else int(math.ceil(math.log2(k)))

    # Integer ids for each x value
    ids = np.fromiter((mapping_id[v] for v in x_list), dtype=np.int64, count=n)

    # Convert ids -> bit matrix (MSB first)
    # Example bits=3: id=5 (101) => [1,0,1]
    shifts = np.arange(bits - 1, -1, -1, dtype=np.int64)
    codes = ((ids[:, None] >> shifts[None, :]) & 1).astype(float)

    return codes

def data_to_binary(data):
    binary_data = []
    for dim in range(data.shape[1]):
        binary_data.append(array_to_binary(data[:,dim]))

    return np.concatenate(binary_data, axis=1)


# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Parameters
ds = 0.2
S = 1.0  # Range for s: [-S, S]
ds_plot = ds / 100  # Interval for plotting
predictor_layers = [500]*5

# Values of A to test
A_values = [ds, S/2, S]
output_dims = [20] # np.linspace(5, 20, 2).astype(int)
input_dim = 3
lr = 0.00001
n_epochs = 10000
optimizer_fn = torch.optim.Adam

def poly(x):
    return x**4 + x**2

def generate_twisted_polynomial_vector(x, input_dim=100, min_deg=5, max_deg=10):
    """
    Generates a high-dimensional vector based on x using high-degree polynomials.
    
    Args:
        x (float or np.array): Scalar input in range [-1, 1].
        input_dim (int): The dimension of the output vector.
        min_deg (int): The lowest degree polynomial to use (prevents linearity).
        max_deg (int): The highest degree polynomial to use.
        
    Returns:
        np.array: A high-dimensional vector (or matrix if x is an array).
    """
    # 1. Generate the Vandermonde matrix for Chebyshev polynomials
    # This calculates T_0(x), T_1(x), ... T_max_deg(x)
    # Shape: (n_samples, max_deg + 1)
    polys = chebvander(x, max_deg)
    
    # 2. Slice to keep only high frequencies
    # We discard degrees 0 to min_deg-1 to ensure no monotonic trends exist
    high_freq_polys = polys[:, min_deg:max_deg+1]
    
    # 3. Create a random mixing matrix (fixed seed for reproducibility)
    # We project the polynomial basis into the desired high dimension
    rng = np.random.RandomState(42)
    feature_dim = high_freq_polys.shape[1]
    W = rng.randn(feature_dim, input_dim)
    
    # 4. Mix the polynomials
    high_dim_vector = high_freq_polys @ W
    
    return high_dim_vector

def f(s, v, a=None):
    if a is None:
        a = np.zeros_like(s)        
    x = (s + a)[:,None]@v
    x_periodic = 2*np.sin(2 * 2*np.pi*(s + a)[:,None]@v)
    # return np.tanh(x_periodic)
    # return 0.1*poly(x_periodic)
    # return poly(x)
    return generate_twisted_polynomial_vector(s + a, input_dim=v.shape[1])

# Define the neural network
class MixedActivationLayer(nn.Module):
    def __init__(self, layer_dim, base_activation=nn.GELU()):
        super().__init__()
        # layer_dim: number of neurons in this layer
        self.layer_dim = layer_dim

        # Determine how many neurons per activation
        num_quarter = layer_dim // 4
        num_half = (layer_dim - 2 * num_quarter)

        # Make indices for identity, x^2, and base activation
        self.id_indices = torch.arange(0, num_quarter)
        self.square_indices = torch.arange(num_quarter, num_quarter * 2)
        self.base_indices = torch.arange(num_quarter * 2, layer_dim)

        self.base_activation = base_activation

    def forward(self, x):
        # x shape: [batch, layer_dim]
        out = torch.zeros_like(x)
        # Identity for first quarter
        out[:, self.id_indices] = x[:, self.id_indices]
        # x^2 for second quarter
        out[:, self.square_indices] = x[:, self.square_indices]**2
        # Base activation for rest
        out[:, self.base_indices] = self.base_activation(x[:, self.base_indices])
        return out

class DeepFCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, mixed_activation=False, base_activation_cls=nn.ReLU):
        super(DeepFCN, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if mixed_activation:
                # Use mixed activations: identity, x^2, base
                layers.append(MixedActivationLayer(hidden_dim, base_activation=base_activation_cls()))
            else:
                # Use standard activation for all
                layers.append(base_activation_cls())
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


def generate_training_data(A, S, ds, vin, vout):
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
    function_samples = f(s_flat, v=vin)
    
    # Compute targets: sin(s + a)
    targets = f(s_flat, v=vout, a=a_flat)
    
    # Input features: [sin(2π * s), a]
    inputs = np.column_stack([function_samples, a_flat])
    
    return inputs, targets, s_flat, a_flat

def train_model(A, S, ds, vin, vout, num_epochs=10, batch_size=64, learning_rate=0.001):
    """Train a deep fully connected network."""
    # Generate training data
    inputs, targets, s_vals, a_vals = generate_training_data(A, S, ds, vin, vout)
    
    # Convert to tensors and move to device
    X = torch.FloatTensor(inputs).to(device)
    y = torch.FloatTensor(targets).to(device)
    
    # Create model and move to device
    model = DeepFCN(input_dim=X.shape[1], hidden_dims=predictor_layers, output_dim=y.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
    
    # Track training loss
    train_losses = []
    
    # Training loop
    model.train()
    for epoch in tqdm(range(num_epochs), desc=f"Training (A={A})"):
        # Forward pass (full batch)
        outputs, hidden = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss for this epoch
        train_losses.append(loss.item())
    
    return model, s_vals, a_vals, inputs, targets, train_losses

def train_hidden_to_target_model(original_model, X, y, vin, vout, hidden_dim=1000, num_epochs=10000, batch_size=256, learning_rate=0.001):
    """Train a single hidden layer model to map hidden activations to targets."""
    # Get hidden activations from the original model
    inputs, targets, s_vals, a_vals = generate_training_data(1e-10, S, ds, vin, vout)
    X = torch.FloatTensor(inputs).to(device)
    y = torch.FloatTensor(targets).to(device)
    inputs_test, targets_test, s_vals_test, a_vals_test = generate_training_data(1e-10, S, ds/100, vin, vout)
    X_test = torch.FloatTensor(inputs_test).to(device)
    y_test = torch.FloatTensor(targets_test).to(device)

    original_model.eval()
    with torch.no_grad():
        _, hidden_activations = original_model(X)
        _, hidden_activations_test = original_model(X_test)
    
    # Get the hidden dimension size
    hidden_size = hidden_activations.shape[1]
    output_dim = y.shape[1]
    
    # Create the new model
    hidden_model = SingleHiddenLayerModel(input_dim=hidden_size, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(hidden_model.parameters(), lr=learning_rate)
    
    # Track training loss
    hidden_train_losses = []
    hidden_test_losses = []
    
    # Training loop
    hidden_model.train()
    for epoch in tqdm(range(num_epochs), desc="Training hidden-to-target model"):
        # Shuffle data
        hidden_model.train()
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
        hidden_train_losses.append(avg_loss)
    
        if epoch // (num_epochs/100) == 0:
            hidden_model.eval()
            with torch.no_grad():
                final_outputs = hidden_model(hidden_activations_test)
                final_mse = criterion(final_outputs, y_test).item()
                hidden_test_losses.append(final_mse)
    
    print(f"Hidden-to-target model final MSE: {final_mse:.6f}")
    
    return hidden_model, final_mse, hidden_train_losses, hidden_test_losses

def plot_all_results(all_results, S, ds_plot, vin, vout):
    """Plot results from all A values in a single figure."""
    num_A = len(all_results)
    
    # Create figure with subplots: N rows (one per A), 6 columns (predictions, predictions2, loss, PCA hidden, PCA targets, hidden-to-target loss)
    n_plots = 6
    fig, axes = plt.subplots(num_A, n_plots, figsize=(4*n_plots, 3*num_A))#, sharey='col')
    
    # Handle case where there's only one A value
    if num_A == 1:
        axes = axes.reshape(1, -1)
    
    # Generate plot data once (same for all A values)
    inputs_plot, targets_plot, s_vals_plot, a_vals_plot = generate_training_data(ds*3, S, ds/100, vin, vout)
    
    # Compute color values: s_vals_plot + a_vals_plot
    color_values = s_vals_plot + a_vals_plot
    
    for idx, (A, model, s_train, a_train, inputs_train, targets_train, train_losses, hidden_train_losses, hidden_test_losses) in enumerate(all_results):
        # Get predictions from model
        model.eval()
        with torch.no_grad():
            X_plot = torch.FloatTensor(inputs_plot).to(device)
            predictions, hidden = model(X_plot)
            predictions = predictions.cpu().numpy()
            hidden = hidden.cpu().numpy()
        
        inputs_binary = data_to_binary(inputs_train)
        latent_binary = data_to_binary((s_train+a_train)[:,None])
        if inputs_binary.shape[1] > 20:
            degree = 0
            degree_latent = 0
        else:
            f_hat_l = [FW_transform(inputs_binary, targets_train[:,i]) for i in range(targets_train.shape[1])]
            degree_l = [calc_degree(f_hat) for f_hat in f_hat_l]
            degree = np.mean(degree_l)
            f_hat_l = [FW_transform(latent_binary, targets_train[:,i]) for i in range(targets_train.shape[1])]
            degree_l = [calc_degree(f_hat) for f_hat in f_hat_l]
            degree_latent = np.mean(degree_l)
        # Calculate MSE
        mse = np.mean((predictions - targets_plot)**2)
        
        # Filter training samples where a=0 (for plotting)
        mask_a_zero = np.abs(a_train) < 1e-10  # Find samples where a ≈ 0
        s_train_a0 = s_train[mask_a_zero]
        targets_train_a0 = targets_train[mask_a_zero, 0]
        
        # Left subplot: Function predictions
        ax_i = 0
        ax_pred = axes[idx, ax_i]
        ax_pred.plot(s_vals_plot, targets_plot[:,0], 'b-', label='True: f(x, 0)')
        ax_pred.plot(s_vals_plot, predictions[:,0], 'r--', label='Predicted', alpha=0.7)
        ax_pred.scatter(s_train_a0, targets_train_a0, c='green', s=30, alpha=0.6, 
                        label='Training samples (a=0)', zorder=5)
        ax_pred.set_xlabel('x', fontsize=6)
        ax_pred.set_ylabel(f'degree={degree}/{inputs_binary.shape[1]}, latent_degree={degree_latent}/{latent_binary.shape[1]}', fontsize=6)
        ax_pred.set_title(f'A={A:.3f}, MSE={mse:.6f}', fontsize=7)
        ax_pred.grid(True, alpha=0.3)
        
        # Left subplot: Function predictions
        targets_train_a0 = targets_train[mask_a_zero, 1]
        ax_i += 1
        ax_pred = axes[idx, ax_i]
        ax_pred.plot(s_vals_plot, targets_plot[:,1], 'b-', label='True: f(x, 0)')
        ax_pred.plot(s_vals_plot, predictions[:,1], 'r--', label='Predicted', alpha=0.7)
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
        ax_loss.plot(epochs, train_losses)
        ax_loss.set_xlabel('Epoch', fontsize=6)
        ax_loss.set_ylabel('Training Loss (MSE)', fontsize=6)
        ax_loss.set_title(f'Training Loss Curve (A={A:.3f})', fontsize=7)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale('log')  # Use log scale for better visualization
        
        # Third subplot: PCA of hidden activities
        # Compute color values: s_vals_plot
        color_values = s_vals_plot
        ax_i += 1
        ax_pca_hidden = axes[idx, ax_i]
        pca_hidden = PCA(n_components=2)
        hidden_pca = pca_hidden.fit_transform(hidden)
        # Calculate R² between PCs and s+a
        reg_hidden = LinearRegression()
        reg_hidden.fit(hidden_pca[:,:3], color_values)
        r2_hidden = r2_score(color_values, reg_hidden.predict(hidden_pca[:,:3]))
        scatter_hidden = ax_pca_hidden.scatter(hidden_pca[:, 0], hidden_pca[:, 1], 
                                                c=color_values, cmap='viridis', 
                                                s=10, alpha=0.6)
        ax_pca_hidden.set_xlabel(f'PC1 ({pca_hidden.explained_variance_ratio_[0]:.2%} variance)', fontsize=6)
        ax_pca_hidden.set_ylabel(f'PC2 ({pca_hidden.explained_variance_ratio_[1]:.2%} variance)', fontsize=6)
        ax_pca_hidden.set_title(f'colored by s: R²={r2_hidden:.3f}', fontsize=7)
        ax_pca_hidden.axis('equal')
        plt.colorbar(scatter_hidden, ax=ax_pca_hidden, label='s+a')
        ax_pca_hidden.grid(True, alpha=0.3)
        
        # Third subplot: PCA of hidden activities colored by actions
        # Compute color values: a_vals_plot
        color_values = a_vals_plot
        ax_i += 1
        ax_pca_hidden = axes[idx, ax_i]
        pca_hidden = PCA(n_components=2)
        hidden_pca = pca_hidden.fit_transform(hidden)
        # Calculate R² between PCs and s+a
        reg_hidden = LinearRegression()
        reg_hidden.fit(hidden_pca[:,:3], color_values)
        r2_hidden = r2_score(color_values, reg_hidden.predict(hidden_pca[:,:3]))
        scatter_hidden = ax_pca_hidden.scatter(hidden_pca[:, 0], hidden_pca[:, 1], 
                                                c=color_values, cmap='viridis', 
                                                s=10, alpha=0.6)
        ax_pca_hidden.set_xlabel(f'PC1 ({pca_hidden.explained_variance_ratio_[0]:.2%} variance)', fontsize=6)
        ax_pca_hidden.set_ylabel(f'PC2 ({pca_hidden.explained_variance_ratio_[1]:.2%} variance)', fontsize=6)
        ax_pca_hidden.set_title(f'colored by a: R²={r2_hidden:.3f}', fontsize=7)
        ax_pca_hidden.axis('equal')
        plt.colorbar(scatter_hidden, ax=ax_pca_hidden, label='s+a')
        ax_pca_hidden.grid(True, alpha=0.3)
        
        # Third subplot: PCA of hidden activities colored by s+a
        # Compute color values: a_vals_plot + s_vals_plot
        color_values = s_vals_plot + a_vals_plot
        ax_i += 1
        ax_pca_hidden = axes[idx, ax_i]
        pca_hidden = PCA(n_components=2)
        hidden_pca = pca_hidden.fit_transform(hidden)
        # Calculate R² between PCs and s+a
        reg_hidden = LinearRegression()
        reg_hidden.fit(hidden_pca[:,:3], color_values)
        r2_hidden = r2_score(color_values, reg_hidden.predict(hidden_pca[:,:3]))
        scatter_hidden = ax_pca_hidden.scatter(hidden_pca[:, 0], hidden_pca[:, 1], 
                                                c=color_values, cmap='viridis', 
                                                s=10, alpha=0.6)
        ax_pca_hidden.set_xlabel(f'PC1 ({pca_hidden.explained_variance_ratio_[0]:.2%} variance)', fontsize=6)
        ax_pca_hidden.set_ylabel(f'PC2 ({pca_hidden.explained_variance_ratio_[1]:.2%} variance)', fontsize=6)
        ax_pca_hidden.set_title(f'colored by s+a: R²={r2_hidden:.3f}', fontsize=7)
        ax_pca_hidden.axis('equal')
        plt.colorbar(scatter_hidden, ax=ax_pca_hidden, label='s+a')
        ax_pca_hidden.grid(True, alpha=0.3)
        
        # # Fourth subplot: PCA of targets
        # ax_i += 1
        # ax_pca_targets = axes[idx, ax_i]
        # pca_targets = PCA(n_components=2)
        # targets_pca = pca_targets.fit_transform(targets_plot)
        # # Calculate R² between PCs and s+a
        # reg_targets = LinearRegression()
        # reg_targets.fit(targets_pca, color_values)
        # r2_targets = r2_score(color_values, reg_targets.predict(targets_pca))
        # scatter_targets = ax_pca_targets.scatter(targets_pca[:, 0], targets_pca[:, 1], 
        #                                           c=color_values, cmap='viridis', 
        #                                           s=10, alpha=0.6)
        # ax_pca_targets.set_xlabel(f'PC1 ({pca_targets.explained_variance_ratio_[0]:.2%} variance)', fontsize=6)
        # ax_pca_targets.set_ylabel(f'PC2 ({pca_targets.explained_variance_ratio_[1]:.2%} variance)', fontsize=6)
        # ax_pca_targets.set_title(f'PCA of Targets (A={A:.3f}, R²={r2_targets:.3f})', fontsize=7)
        # plt.colorbar(scatter_targets, ax=ax_pca_targets, label='s+a')
        # ax_pca_targets.grid(True, alpha=0.3)
        
        # # Fifth subplot: Hidden-to-target training loss curve
        # ax_i += 1
        # ax_hidden_loss = axes[idx, ax_i]
        # hidden_epochs = np.arange(1, len(hidden_train_losses) + 1)
        # hidden_epochs_test = np.linspace(1, len(hidden_train_losses), len(hidden_test_losses))
        # ax_hidden_loss.plot(hidden_epochs, hidden_train_losses, 'c-', label='Training')
        # ax_hidden_loss.plot(hidden_epochs_test, hidden_test_losses, 'c--', label='Test')
        # ax_hidden_loss.legend(fontsize=5.5)
        # ax_hidden_loss.set_xlabel('Epoch', fontsize=6)
        # ax_hidden_loss.set_ylabel('Hidden-to-Target Loss (MSE)', fontsize=6)
        # ax_hidden_loss.set_title(f'Hidden-to-Target Loss (A={A:.3f})', fontsize=7)
        # ax_hidden_loss.grid(True, alpha=0.3)
        # ax_hidden_loss.set_yscale('log')  # Use log scale for better visualization
        
        # Print error statistics
        mae = np.mean(np.abs(predictions - targets_plot))
        print(f"\nA={A:.3f} - MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    plt.tight_layout()
    return fig

def plot_pca_grid(all_results_2d, S, ds, vin, vout):
    """Plot PCA of hidden activations in a grid: rows = output_dims, columns = A values."""
    num_output_dims = len(all_results_2d)
    num_A = len(all_results_2d[0]) if num_output_dims > 0 else 0
    
    # Create figure with subplots: rows = output_dims, columns = A values
    fig, axes = plt.subplots(num_output_dims, num_A, figsize=(4*num_A, 3*num_output_dims))
    
    # Handle case where there's only one output_dim or one A value
    if num_output_dims == 1:
        axes = axes.reshape(1, -1)
    elif num_A == 1:
        axes = axes.reshape(-1, 1)
    
    # Generate plot data once per output_dim (since vout changes with output_dim)
    for row_idx, all_results in enumerate(all_results_2d):
        # Get output_dim from the first result (we need to infer it from the model)
        # Actually, we need vout for this output_dim - but we don't have it stored
        # Let's regenerate it using the same seed logic
        output_dim = output_dims[row_idx]
        np.random.seed(41)
        vin_current = np.random.randn(input_dim)[None,:]
        vout_current = np.random.randn(output_dim)[None,:]
        
        # Generate plot data for this output_dim
        inputs_plot, targets_plot, s_vals_plot, a_vals_plot = generate_training_data(1e-10, S, ds/100, vin_current, vout_current)
        
        # Compute color values: s_vals_plot + a_vals_plot
        color_values = s_vals_plot + a_vals_plot
        
        for col_idx, (A, model, s_train, a_train, inputs_train, targets_train, train_losses, hidden_train_losses, hidden_test_losses) in enumerate(all_results):
            # Get hidden activations from model
            model.eval()
            with torch.no_grad():
                X_plot = torch.FloatTensor(inputs_plot).to(device)
                predictions, hidden = model(X_plot)
                hidden = hidden.cpu().numpy()
            
            # Compute PCA of hidden activations
            ax_pca_hidden = axes[row_idx, col_idx]
            pca_hidden = PCA(n_components=2)
            hidden_pca = pca_hidden.fit_transform(hidden)
            
            # Calculate R² between PC1 and s+a
            reg_hidden = LinearRegression()
            reg_hidden.fit(hidden_pca[:,[0]], color_values)
            r2_hidden = r2_score(color_values, reg_hidden.predict(hidden_pca[:,[0]]))
            
            # Plot PCA scatter
            scatter_hidden = ax_pca_hidden.scatter(hidden_pca[:, 0], hidden_pca[:, 1], 
                                                    c=color_values, cmap='viridis', 
                                                    s=10, alpha=0.6)
            ax_pca_hidden.set_xlabel(f'PC1 ({pca_hidden.explained_variance_ratio_[0]:.2%})', fontsize=8)
            ax_pca_hidden.set_ylabel(f'PC2 ({pca_hidden.explained_variance_ratio_[1]:.2%})', fontsize=8)
            
            # Set title with A value and R²
            title_parts = []
            if row_idx == 0:
                title_parts.append(f'A={A:.3f}')
            if col_idx == 0:
                title_parts.append(f'out_dim={output_dim}')
            title_parts.append(f'R²={r2_hidden:.3f}')
            ax_pca_hidden.set_title(', '.join(title_parts), fontsize=8)
            
            ax_pca_hidden.axis('equal')
            plt.colorbar(scatter_hidden, ax=ax_pca_hidden, label='s+a')
            ax_pca_hidden.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Main execution
if __name__ == "__main__":
    # Collect all results first
    all_results_2d = []
    for output_dim in output_dims:
        all_results = []
        for A in A_values:
            print(f"\n{'='*60}")
            print(f"Processing A = {A}")
            print(f"{'='*60}")
            
            np.random.seed(41)
            vin = np.random.randn(input_dim)[None,:]
            vout = np.random.randn(output_dim)[None,:]
            
            # Train model
            model, s_train, a_train, inputs_train, targets_train, train_losses = train_model(A, S, ds, vin, vout, num_epochs=n_epochs, learning_rate=lr)
            
            # Prepare data for hidden-to-target model training
            inputs, targets, s_vals, a_vals = generate_training_data(A, S, ds, vin, vout)
            X = torch.FloatTensor(inputs).to(device)
            y = torch.FloatTensor(targets).to(device)
            
            # Train hidden-to-target model
            hidden_model, hidden_mse, hidden_train_losses, hidden_test_losses = train_hidden_to_target_model(
                model, X, y, vin, vout, 
                hidden_dim=1000, 
                num_epochs=1000, 
                learning_rate=0.001
            )
            
            # Store results
            all_results.append((A, model, s_train, a_train, inputs_train, targets_train, train_losses, hidden_train_losses, hidden_test_losses))
            
            print(f"Completed processing for A = {A}\n")
        
        all_results_2d.append(all_results)
        
        # Plot all results in a single figure
        fig = plot_all_results(all_results, S, ds_plot, vin, vout)
        plt.savefig(f'plot_all_A_values_{output_dim}.png', dpi=150)
        plt.close()

    # Plot PCA grid
    fig = plot_pca_grid(all_results_2d, S, ds, vin, vout)
    plt.savefig('plot_pca_grid.png', dpi=150)
    plt.close()