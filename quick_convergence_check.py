import numpy as np
import matplotlib.pyplot as plt
import os

def quick_convergence_check(loss_arrays, names=None, recent_epochs=100, threshold=1e-4):
    """
    Quick convergence check for multiple training runs.
    
    Args:
        loss_arrays: List of loss arrays from different training runs
        names: List of names for each training run (optional)
        recent_epochs: Number of recent epochs to analyze
        threshold: Threshold for considering loss change significant
    
    Returns:
        dict: Quick convergence assessment for each run
    """
    
    if names is None:
        names = [f"Training {i+1}" for i in range(len(loss_arrays))]
    
    results = {}
    
    print("Quick Convergence Assessment:")
    print("=" * 50)
    
    for loss_array, name in zip(loss_arrays, names):
        if len(loss_array) < recent_epochs:
            print(f"{name}: Insufficient data ({len(loss_array)} epochs)")
            continue
        
        # Get recent loss values
        recent_losses = loss_array[-recent_epochs:]
        
        # Calculate metrics
        final_loss = loss_array[-1]
        loss_change = abs(loss_array[-1] - loss_array[-recent_epochs]) / loss_array[-recent_epochs]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        cv = loss_std / loss_mean if loss_mean > 0 else np.inf
        
        # Determine status
        if loss_change < threshold and cv < 0.1:
            status = "CONVERGED"
            color = "green"
        elif loss_change < threshold * 10:
            status = "LIKELY CONVERGED"
            color = "orange"
        elif loss_change > 0.01:
            status = "STILL TRAINING"
            color = "red"
        else:
            status = "UNCLEAR"
            color = "gray"
        
        results[name] = {
            'final_loss': final_loss,
            'loss_change': loss_change,
            'coefficient_of_variation': cv,
            'status': status,
            'recent_std': loss_std
        }
        
        print(f"{name}:")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Loss Change: {loss_change:.2e}")
        print(f"  CV: {cv:.3f}")
        print(f"  Status: {status}")
        print()
    
    return results

def plot_convergence_comparison(loss_arrays, names=None, save_dir="figures"):
    """Plot all loss curves for visual comparison."""
    
    if names is None:
        names = [f"Training {i+1}" for i in range(len(loss_arrays))]
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Plot all loss curves
    for i, (loss_array, name) in enumerate(zip(loss_arrays, names)):
        plt.plot(loss_array, label=name, alpha=0.8, linewidth=1.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss comparison plot saved to {os.path.join(save_dir, 'loss_comparison.png')}")

def convergence_summary_table(results):
    """Create a summary table of convergence results."""
    
    print("\nConvergence Summary Table:")
    print("=" * 80)
    print(f"{'Training Run':<20} {'Final Loss':<12} {'Loss Change':<12} {'CV':<8} {'Status':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<20} {result['final_loss']:<12.6f} {result['loss_change']:<12.2e} "
              f"{result['coefficient_of_variation']:<8.3f} {result['status']:<15}")
    
    print("-" * 80)

# Example usage with your loss arrays
def analyze_your_training_runs():
    """Example of how to use this with your actual loss arrays."""
    
    # Replace these with your actual loss arrays
    # Example: loss_arrays = [losses1, losses2, losses3, ...]
    
    # Generate example data for demonstration
    np.random.seed(42)
    epochs = 1000
    
    # Different types of training runs
    loss1 = 1.0 * np.exp(-np.arange(epochs) / 200) + 0.1 + 0.01 * np.random.randn(epochs)  # Converged
    loss2 = 1.0 * np.exp(-np.arange(epochs) / 500) + 0.2 + 0.02 * np.random.randn(epochs)  # Still training
    loss3 = 0.5 + 0.3 * np.exp(-np.arange(epochs) / 300) + 0.1 * np.sin(np.arange(epochs) / 50) + 0.05 * np.random.randn(epochs)  # Oscillating
    loss4 = 2.0 * np.exp(-np.arange(epochs) / 800) + 0.05 + 0.005 * np.random.randn(epochs)  # Slow convergence
    
    loss_arrays = [loss1, loss2, loss3, loss4]
    names = ['Fast Convergence', 'Slow Training', 'Oscillating', 'Very Slow']
    
    # Quick convergence check
    results = quick_convergence_check(loss_arrays, names)
    
    # Plot comparison
    plot_convergence_comparison(loss_arrays, names)
    
    # Summary table
    convergence_summary_table(results)
    
    return results

if __name__ == "__main__":
    results = analyze_your_training_runs() 