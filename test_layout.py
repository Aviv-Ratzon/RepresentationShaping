#!/usr/bin/env python3
"""
Test script to verify the layout fixes are working correctly.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set font to avoid Times New Roman issues
plt.rcParams['font.family'] = 'DejaVu Serif'

def test_layout():
    """Test the layout with simple data."""
    
    # Create test data
    np.random.seed(42)
    n_points = 100
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (title, data) in enumerate([
        ('Test 1', np.random.randn(n_points, 2)),
        ('Test 2', np.random.randn(n_points, 2) * 0.5),
        ('Test 3', np.random.randn(n_points, 2) * 2)
    ]):
        ax = axes[i]
        
        # Create scatter plot
        scatter = ax.scatter(data[:, 0], data[:, 1], 
                           c=np.random.randint(0, 10, n_points), 
                           cmap='viridis', alpha=0.7, s=20)
        
        ax.set_title(f'$\mathbf{{{title}}}$', fontsize=14, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Set consistent limits
        ax.set_xlim(data[:, 0].min() * 1.1, data[:, 0].max() * 1.1)
        ax.set_ylim(data[:, 1].min() * 1.1, data[:, 1].max() * 1.1)
    
    # Add colorbar to the rightmost subplot
    cbar = fig.colorbar(scatter, ax=axes[-1], shrink=0.8, aspect=30)
    cbar.set_label('Target Label', fontsize=12)
    
    # Add overall title
    fig.suptitle('Layout Test - Square Boundaries and Equal Scaling', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Use subplots_adjust for better control
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.92, wspace=0.3)
    
    # Save the test plot
    output_path = "layout_test.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Layout test plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    test_layout()

