"""
Test and visualize participation ratio for high-dimensional nonlinear function dataset.

This script creates a dataset where:
- Input: concatenated tuples {f(s), a} where s is latent variable, f is nonlinear function, a ∈ [-A, A]
- Target: f(s+a)
- We analyze the participation ratio of (X^TX)^-1X^TY as a function of A
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class NonlinearFunctionDataset:
    """Dataset generator for high-dimensional nonlinear functions."""
    
    def __init__(self, 
                 dim: int = 100,
                 n_samples: int = 1000,
                 s_range: Tuple[float, float] = (-2, 2),
                 A_range: Tuple[float, float] = (0.1, 2.0),
                 n_A_points: int = 20):
        """
        Initialize the dataset generator.
        
        Args:
            dim: Dimensionality of the nonlinear function output
            n_samples: Number of samples to generate
            s_range: Range for the latent variable s
            A_range: Range for the A parameter (controls perturbation magnitude)
            n_A_points: Number of A values to test
        """
        self.dim = dim
        self.n_samples = n_samples
        self.s_range = s_range
        self.A_range = A_range
        self.n_A_points = n_A_points
        
    def nonlinear_function(self, s: np.ndarray, continuous: bool = True) -> np.ndarray:
        """
        Generate high-dimensional nonlinear function output.
        
        Args:
            s: Latent variable values
            continuous: If True, use continuous function; if False, use piecewise continuous
            
        Returns:
            Array of shape (len(s), self.dim) containing function outputs
        """
        s = np.atleast_1d(s)
        n = len(s)
        
        if continuous:
            # Continuous nonlinear function: combination of trigonometric and polynomial terms
            f = np.zeros((n, self.dim))
            for i in range(self.dim):
                # Different frequency and phase for each dimension
                freq = 1 + 0.1 * i
                phase = 0.5 * i
                f[:, i] = (np.sin(freq * s + phase) + 
                          0.5 * np.cos(2 * freq * s + phase) + 
                          0.1 * s**2 + 
                          0.05 * s**3)
        else:
            # Piecewise continuous function
            f = np.zeros((n, self.dim))
            for i in range(self.dim):
                # Different breakpoints for each dimension
                breakpoint = -1 + 2 * i / self.dim
                mask1 = s < breakpoint
                mask2 = s >= breakpoint
                
                f[mask1, i] = np.sin(s[mask1] + i) + 0.5 * s[mask1]**2
                f[mask2, i] = np.cos(s[mask2] + i) + 0.3 * s[mask2]**3
        
        return f
    
    def generate_data(self, A: float, continuous: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dataset for given A value.
        
        Args:
            A: Perturbation magnitude parameter
            continuous: Whether to use continuous or piecewise continuous function
            
        Returns:
            X: Input data of shape (n_samples, dim + 1) - concatenated {f(s), a}
            Y: Target data of shape (n_samples, dim) - f(s+a)
        """
        # Generate latent variables s
        s = np.random.uniform(self.s_range[0], self.s_range[1], self.n_samples)
        
        # Generate perturbation values a
        a = np.random.uniform(-A, A, self.n_samples)
        
        # Generate function values f(s)
        f_s = self.nonlinear_function(s, continuous)
        
        # Generate target values f(s+a)
        s_perturbed = s + a
        f_s_plus_a = self.nonlinear_function(s_perturbed, continuous)
        
        # Create input data: concatenate f(s) and a
        X = np.column_stack([f_s, a])
        
        return X, f_s_plus_a
    
    def compute_participation_ratio(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute participation ratio of (X^TX)^-1X^TY.
        
        Args:
            X: Input data matrix
            Y: Target data matrix
            
        Returns:
            Participation ratio (sum of squared singular values)^2 / sum of 4th powers
        """
        # Compute (X^TX)^-1X^TY
        try:
            # Use pseudo-inverse for numerical stability
            X_pinv = np.linalg.pinv(X)
            W = X_pinv @ Y
        except np.linalg.LinAlgError:
            # Fallback to SVD-based pseudo-inverse using numpy
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            s_inv = np.where(s > 1e-10, 1/s, 0)
            X_pinv = Vt.T @ np.diag(s_inv) @ U.T
            W = X_pinv @ Y
        
        # Compute SVD of W
        U_w, s_w, Vt_w = np.linalg.svd(W, full_matrices=False)
        
        # Participation ratio: (sum of squared singular values)^2 / sum of 4th powers
        sum_squared = np.sum(s_w**2)
        sum_fourth = np.sum(s_w**4)
        
        if sum_fourth > 1e-12:
            participation_ratio = sum_squared**2 / sum_fourth
        else:
            participation_ratio = 1.0  # All singular values are effectively zero
        
        return participation_ratio
    
    def sweep_A_parameter(self, continuous: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep over A parameter and compute participation ratios.
        
        Args:
            continuous: Whether to use continuous or piecewise continuous function
            
        Returns:
            A_values: Array of A values tested
            participation_ratios: Corresponding participation ratios
        """
        A_values = np.linspace(self.A_range[0], self.A_range[1], self.n_A_points)
        participation_ratios = np.zeros(self.n_A_points)
        
        print(f"Computing participation ratios for {'continuous' if continuous else 'piecewise'} function...")
        
        for i, A in enumerate(A_values):
            X, Y = self.generate_data(A, continuous)
            participation_ratios[i] = self.compute_participation_ratio(X, Y)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{self.n_A_points} A values")
        
        return A_values, participation_ratios

def plot_results(A_values_cont: np.ndarray, 
                participation_ratios_cont: np.ndarray,
                A_values_piece: np.ndarray, 
                participation_ratios_piece: np.ndarray,
                dim: int,
                n_samples: int):
    """Plot participation ratio results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot continuous function results
    ax1.plot(A_values_cont, participation_ratios_cont, 'o-', linewidth=2, markersize=6, 
             label='Continuous Function', color='blue')
    ax1.set_xlabel('A (Perturbation Magnitude)', fontsize=12)
    ax1.set_ylabel('Participation Ratio', fontsize=12)
    ax1.set_title(f'Participation Ratio vs A\n(Continuous Function, dim={dim}, n_samples={n_samples})', 
                  fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot piecewise continuous function results
    ax2.plot(A_values_piece, participation_ratios_piece, 's-', linewidth=2, markersize=6,
             label='Piecewise Continuous Function', color='red')
    ax2.set_xlabel('A (Perturbation Magnitude)', fontsize=12)
    ax2.set_ylabel('Participation Ratio', fontsize=12)
    ax2.set_title(f'Participation Ratio vs A\n(Piecewise Continuous Function, dim={dim}, n_samples={n_samples})', 
                  fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Combined plot
    plt.figure(figsize=(10, 6))
    plt.plot(A_values_cont, participation_ratios_cont, 'o-', linewidth=2, markersize=6,
             label='Continuous Function', color='blue')
    plt.plot(A_values_piece, participation_ratios_piece, 's-', linewidth=2, markersize=6,
             label='Piecewise Continuous Function', color='red')
    plt.xlabel('A (Perturbation Magnitude)', fontsize=12)
    plt.ylabel('Participation Ratio', fontsize=12)
    plt.title(f'Participation Ratio Comparison\n(dim={dim}, n_samples={n_samples})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_singular_values(A_values: np.ndarray, 
                          participation_ratios: np.ndarray,
                          dataset: NonlinearFunctionDataset,
                          continuous: bool = True,
                          n_examples: int = 3):
    """Analyze singular value distributions for selected A values."""
    
    # Select A values to analyze (low, medium, high)
    indices = np.linspace(0, len(A_values)-1, n_examples, dtype=int)
    selected_A = A_values[indices]
    
    fig, axes = plt.subplots(1, n_examples, figsize=(15, 5))
    if n_examples == 1:
        axes = [axes]
    
    for i, A in enumerate(selected_A):
        X, Y = dataset.generate_data(A, continuous)
        
        # Compute W = (X^TX)^-1X^TY
        try:
            X_pinv = np.linalg.pinv(X)
            W = X_pinv @ Y
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            s_inv = np.where(s > 1e-10, 1/s, 0)
            X_pinv = Vt.T @ np.diag(s_inv) @ U.T
            W = X_pinv @ Y
        
        # Compute SVD of W
        U_w, s_w, Vt_w = np.linalg.svd(W, full_matrices=False)
        
        # Plot singular values
        axes[i].semilogy(s_w, 'o-', markersize=4)
        axes[i].set_xlabel('Singular Value Index', fontsize=10)
        axes[i].set_ylabel('Singular Value', fontsize=10)
        axes[i].set_title(f'A = {A:.2f}\nPR = {participation_ratios[indices[i]]:.2f}', fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Singular Value Distributions\n({"Continuous" if continuous else "Piecewise"} Function)', 
                 fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the analysis."""
    
    # Parameters
    dim = 5  # Dimensionality of nonlinear function
    n_samples = 1000  # Number of samples
    A_range = (0.01, 1)  # Range for A parameter
    n_A_points = 100  # Number of A values to test
    
    print("=" * 60)
    print("HIGH-DIMENSIONAL NONLINEAR FUNCTION PARTICIPATION RATIO ANALYSIS")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  - Function dimensionality: {dim}")
    print(f"  - Number of samples: {n_samples}")
    print(f"  - A range: {A_range}")
    print(f"  - Number of A points: {n_A_points}")
    print()
    
    # Create dataset generator
    dataset = NonlinearFunctionDataset(
        dim=dim,
        n_samples=n_samples,
        A_range=A_range,
        n_A_points=n_A_points
    )
    
    # Test continuous function
    print("Testing continuous nonlinear function...")
    A_values_cont, participation_ratios_cont = dataset.sweep_A_parameter(continuous=True)
    
    # Test piecewise continuous function
    print("\nTesting piecewise continuous nonlinear function...")
    A_values_piece, participation_ratios_piece = dataset.sweep_A_parameter(continuous=False)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(A_values_cont, participation_ratios_cont,
                A_values_piece, participation_ratios_piece,
                dim, n_samples)
    
    # # Analyze singular value distributions
    # print("\nAnalyzing singular value distributions...")
    # analyze_singular_values(A_values_cont, participation_ratios_cont, 
    #                       dataset, continuous=True)
    # analyze_singular_values(A_values_piece, participation_ratios_piece, 
    #                       dataset, continuous=False)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Continuous Function:")
    print(f"  - Min participation ratio: {np.min(participation_ratios_cont):.3f}")
    print(f"  - Max participation ratio: {np.max(participation_ratios_cont):.3f}")
    print(f"  - Mean participation ratio: {np.mean(participation_ratios_cont):.3f}")
    print(f"  - Std participation ratio: {np.std(participation_ratios_cont):.3f}")
    
    print(f"\nPiecewise Continuous Function:")
    print(f"  - Min participation ratio: {np.min(participation_ratios_piece):.3f}")
    print(f"  - Max participation ratio: {np.max(participation_ratios_piece):.3f}")
    print(f"  - Mean participation ratio: {np.mean(participation_ratios_piece):.3f}")
    print(f"  - Std participation ratio: {np.std(participation_ratios_piece):.3f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
