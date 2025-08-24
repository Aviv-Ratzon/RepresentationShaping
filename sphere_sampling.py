import numpy as np
import torch
import matplotlib.pyplot as plt

def uniform_sphere_sampling(n_points, dimension, radius=1.0, seed=None):
    """
    Uniformly sample points on a sphere in n-dimensional space.
    
    This method uses the Box-Muller transform to generate uniform samples
    on a sphere by sampling from a multivariate normal distribution and
    then normalizing the vectors.
    
    Parameters:
    -----------
    n_points : int
        Number of points to sample
    dimension : int
        Dimension of the sphere (e.g., 18 for 18D sphere)
    radius : float, optional
        Radius of the sphere (default: 1.0)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    points : np.ndarray
        Array of shape (n_points, dimension) with uniformly sampled points
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random points from standard normal distribution
    points = np.random.normal(0, 1, (n_points, dimension))
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Scale by radius
    points = points * radius
    
    return points

def verify_uniformity(points, n_bins=10):
    """
    Verify that points are approximately uniformly distributed on the sphere.
    
    Parameters:
    -----------
    points : np.ndarray
        Array of points on the sphere
    n_bins : int
        Number of bins for histogram analysis
    
    Returns:
    --------
    uniformity_score : float
        Measure of uniformity (closer to 1.0 is more uniform)
    """
    # Calculate angles between points and a reference direction
    reference = np.ones(points.shape[1])
    reference = reference / np.linalg.norm(reference)
    
    angles = np.arccos(np.clip(np.dot(points, reference), -1.0, 1.0))
    
    # Create histogram of angles
    hist, _ = np.histogram(angles, bins=n_bins, density=True)
    
    # Expected uniform density
    expected_density = 1.0 / (np.pi * n_bins)
    
    # Calculate uniformity score (variance from uniform)
    uniformity_score = 1.0 - np.var(hist) / (expected_density ** 2)
    
    return uniformity_score

def sample_sphere_for_analysis(n_points=1000, dimension=18, radius=1.0, seed=42):
    """
    Sample points on a sphere and return them in a format suitable for analysis.
    
    Parameters:
    -----------
    n_points : int
        Number of points to sample
    dimension : int
        Dimension of the sphere
    radius : float
        Radius of the sphere
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing the sampled points and metadata
    """
    # Sample points
    sphere_points = uniform_sphere_sampling(n_points, dimension, radius, seed)
    
    # Verify properties
    norms = np.linalg.norm(sphere_points, axis=1)
    uniformity = verify_uniformity(sphere_points)
    
    # Convert to torch tensor
    sphere_points_tensor = torch.tensor(sphere_points, dtype=torch.float32)
    
    return {
        'points': sphere_points,
        'points_tensor': sphere_points_tensor,
        'n_points': n_points,
        'dimension': dimension,
        'radius': radius,
        'norms': norms,
        'uniformity_score': uniformity,
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms)
    }

def visualize_sphere_sampling_2d(points, title="2D Projection of Sphere Points"):
    """
    Visualize sphere points by projecting to 2D using PCA.
    
    Parameters:
    -----------
    points : np.ndarray
        Points on the sphere
    title : str
        Title for the plot
    """
    from sklearn.decomposition import PCA
    
    # Project to 2D using PCA
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], alpha=0.6, s=20)
    plt.title(title)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print explained variance
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

if __name__ == "__main__":
    # Example usage for 18-dimensional sphere
    print("Sampling points on 18-dimensional sphere...")
    
    result = sample_sphere_for_analysis(n_points=1000, dimension=18, radius=1.0, seed=42)
    
    print(f"Sampled {result['n_points']} points on a {result['dimension']}-dimensional sphere")
    print(f"Points shape: {result['points'].shape}")
    print(f"Mean radius: {result['mean_norm']:.6f} ± {result['std_norm']:.6f}")
    print(f"Uniformity score: {result['uniformity_score']:.4f} (closer to 1.0 is more uniform)")
    
    # Visualize if matplotlib is available
    try:
        visualize_sphere_sampling_2d(result['points'], "18D Sphere Points Projected to 2D")
    except ImportError:
        print("Matplotlib not available for visualization")
