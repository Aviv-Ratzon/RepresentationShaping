import numpy as np
import matplotlib.pyplot as plt
from linear_networks_comparison import generate_multiclass_data
from sklearn.decomposition import PCA
import os

def test_high_dimensional_data():
    """Test the generate_multiclass_data function with different numbers of features"""
    
    # Test different feature dimensions
    feature_dims = [2, 3, 5, 10]
    n_classes = 4
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, n_features in enumerate(feature_dims):
        print(f"Generating data with {n_features} features and {n_classes} classes...")
        
        # Generate data
        X, y, y_onehot = generate_multiclass_data(n_samples=500, n_classes=n_classes, 
                                                 n_features=n_features)
        
        # For high-dimensional data, use PCA to visualize
        if n_features > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            X_plot = X_pca
            title_suffix = f" (PCA projection)"
        else:
            X_plot = X
            title_suffix = ""
        
        # Plot
        ax = axes[idx]
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
        ax.set_title(f'{n_features} Features, {n_classes} Classes{title_suffix}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax)
        
        print(f"  Data shape: {X.shape}")
        print(f"  Unique classes: {np.unique(y)}")
        print(f"  One-hot shape: {y_onehot.shape}")
        
        if n_features > 2:
            explained_var = pca.explained_variance_ratio_
            print(f"  PCA explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")
    
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", 'high_dimensional_data_test.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test with more classes than features
    print("\nTesting with more classes than features...")
    n_features = 3
    n_classes = 5
    
    X, y, y_onehot = generate_multiclass_data(n_samples=500, n_classes=n_classes, 
                                             n_features=n_features)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot first two dimensions
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
    ax1.set_title(f'{n_classes} Classes, {n_features} Features\n(First 2 dimensions)')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot dimensions 1 and 3
    scatter2 = ax2.scatter(X[:, 0], X[:, 2], c=y, cmap='viridis', alpha=0.7, s=50)
    ax2.set_title(f'{n_classes} Classes, {n_features} Features\n(Dimensions 1 & 3)')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 3')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", 'more_classes_than_features_test.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Data shape: {X.shape}")
    print(f"  Unique classes: {np.unique(y)}")
    print(f"  One-hot shape: {y_onehot.shape}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_high_dimensional_data() 