#!/usr/bin/env python3
"""
Script to create a single row PCA comparison plot for regularization experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
import glob

# Set font to avoid Times New Roman issues
plt.rcParams['font.family'] = 'DejaVu Serif'

def format_regularization_title(reg_type):
    """Format regularization type as LaTeX title."""
    if reg_type == 'none':
        return r'$\mathbf{No\ Regularization}$'
    elif reg_type == 'l1':
        return r'$\mathbf{L1\ Regularization}$'
    elif reg_type == 'l2':
        return r'$\mathbf{L2\ Regularization}$'
    elif reg_type == 'dropout':
        return r'$\mathbf{Dropout\ Regularization}$'
    elif reg_type == 'l1_l2':
        return r'$\mathbf{L1+L2\ Regularization}$'
    elif reg_type == 'all':
        return r'$\mathbf{All\ Regularization}$'
    else:
        return f'{reg_type.upper()}'

def load_latent_data(plots_dir):
    """Load latent data from .npz files."""
    npz_files = glob.glob(os.path.join(plots_dir, "latents_*.npz"))
    if not npz_files:
        return None, None
    
    # Get the latest epoch file
    latest_file = max(npz_files, key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
    
    data = np.load(latest_file)
    return data['all_latents'], data['all_targets']

def create_pca_comparison():
    """Create single row PCA comparison plot."""
    
    # Define regularization types and their directories
    reg_types = ['none', 'l1', 'l2', 'dropout', 'l1_l2', 'all']
    base_dir = "MNIST_small"
    
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, len(reg_types), figsize=(5*len(reg_types), 5))
    if len(reg_types) == 1:
        axes = [axes]
    
    # Store all PCA data for consistent colorbar
    all_pca_data = []
    all_targets = []
    
    # First pass: collect all data for consistent colorbar
    for reg_type in reg_types:
        plots_dir = os.path.join(base_dir, f"A_1_{reg_type}", "plots")
        if os.path.exists(plots_dir):
            latents, targets = load_latent_data(plots_dir)
            if latents is not None:
                all_pca_data.append(latents)
                all_targets.append(targets)
    
    # Perform PCA on combined data for consistent scaling
    if all_pca_data:
        combined_latents = np.vstack(all_pca_data)
        combined_targets = np.hstack(all_targets)
        pca_combined = PCA(n_components=2)
        pca_combined.fit(combined_latents)
        
        # Get global min/max for consistent colorbar
        global_min = combined_targets.min()
        global_max = combined_targets.max()
    else:
        print("No latent data found!")
        return
    
    # Second pass: create individual plots
    for i, reg_type in enumerate(reg_types):
        ax = axes[i]
        plots_dir = os.path.join(base_dir, f"A_1_{reg_type}", "plots")
        
        if os.path.exists(plots_dir):
            latents, targets = load_latent_data(plots_dir)
            if latents is not None:
                # Transform using the global PCA
                latents_2d = pca_combined.transform(latents)
                
                # Create scatter plot
                scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                                  c=targets, cmap='viridis', alpha=0.7, s=15,
                                  vmin=global_min, vmax=global_max)
                
                ax.set_title(format_regularization_title(reg_type), fontsize=14, fontweight='bold')
                ax.set_xlabel(f'PC1 ({pca_combined.explained_variance_ratio_[0]:.1%})', fontsize=11)
                ax.set_ylabel(f'PC2 ({pca_combined.explained_variance_ratio_[1]:.1%})', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(latents_2d[:, 0].min() * 1.1, latents_2d[:, 0].max() * 1.1)
                ax.set_ylim(latents_2d[:, 1].min() * 1.1, latents_2d[:, 1].max() * 1.1)
            else:
                ax.text(0.5, 0.5, f'No data\nfor {reg_type}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='red')
                ax.set_title(format_regularization_title(reg_type), fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'Directory not found\nfor {reg_type}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_title(f'{reg_type.upper()}', fontsize=14, fontweight='bold')
    
    # Add single colorbar to the rightmost subplot
    cbar = fig.colorbar(scatter, ax=axes[-1], shrink=0.8, aspect=30)
    cbar.set_label('Target Label', fontsize=12)
    
    # Add overall title
    fig.suptitle('PCA Visualization of Latent Space with Different Regularization Types', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout to prevent overlap and ensure proper spacing
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.92, wspace=0.3)
    
    # Save the combined plot
    output_path = "pca_regularization_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PCA comparison plot saved to: {output_path}")
    
    plt.show()

def main():
    """Main function."""
    print("Creating PCA comparison plot for regularization experiments...")
    
    # Check if regularization experiments exist
    base_dir = "MNIST_small"
    reg_types = ['none', 'l1', 'l2', 'dropout', 'l1_l2', 'all']
    
    existing_experiments = []
    for reg_type in reg_types:
        plots_dir = os.path.join(base_dir, f"A_1_{reg_type}", "plots")
        if os.path.exists(plots_dir):
            existing_experiments.append(reg_type)
    
    if not existing_experiments:
        print("No regularization experiments found!")
        print("Please run the regularization experiments first using:")
        print("  ./run_regularization_experiments.sh")
        return
    
    print(f"Found experiments for: {existing_experiments}")
    
    # Create the comparison plot
    create_pca_comparison()
    
    print("Done!")

if __name__ == "__main__":
    main()
