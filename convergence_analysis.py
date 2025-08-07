import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.linear_model import LinearRegression

def analyze_convergence(loss_arrays, names=None, convergence_threshold=1e-4, window_size=100, 
                       min_epochs=500, plot_results=True, save_dir="figures"):
    """
    Analyze convergence of multiple training loss profiles.
    
    Args:
        loss_arrays: List of loss arrays from different training runs
        names: List of names for each training run (optional)
        convergence_threshold: Threshold for considering loss converged
        window_size: Size of window for slope calculation
        min_epochs: Minimum epochs before considering convergence
        plot_results: Whether to plot the results
        save_dir: Directory to save plots
    
    Returns:
        dict: Analysis results for each training run
    """
    
    if names is None:
        names = [f"Training {i+1}" for i in range(len(loss_arrays))]
    
    results = {}
    
    for idx, (loss_array, name) in enumerate(zip(loss_arrays, names)):
        print(f"\nAnalyzing {name}...")
        
        # Basic statistics
        final_loss = loss_array[-1]
        min_loss = np.min(loss_array)
        max_loss = np.max(loss_array)
        loss_range = max_loss - min_loss
        
        # Calculate convergence metrics
        convergence_metrics = calculate_convergence_metrics(loss_array, window_size, min_epochs)
        
        # Determine convergence status
        convergence_status = determine_convergence_status(
            loss_array, convergence_metrics, convergence_threshold, min_epochs
        )
        
        results[name] = {
            'final_loss': final_loss,
            'min_loss': min_loss,
            'max_loss': max_loss,
            'loss_range': loss_range,
            'convergence_metrics': convergence_metrics,
            'convergence_status': convergence_status,
            'loss_array': loss_array
        }
        
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Loss range: {loss_range:.6f}")
        print(f"  Convergence status: {convergence_status}")
        if convergence_status == 'converged':
            print(f"  Converged at epoch: {convergence_metrics['convergence_epoch']}")
    
    if plot_results:
        plot_convergence_analysis(results, save_dir)
    
    return results

def calculate_convergence_metrics(loss_array, window_size=100, min_epochs=500):
    """Calculate various convergence metrics for a loss array."""
    
    metrics = {}
    
    # Calculate moving average and slope
    if len(loss_array) >= window_size:
        moving_avg = np.convolve(loss_array, np.ones(window_size)/window_size, mode='valid')
        epochs = np.arange(window_size, len(loss_array) + 1)
        
        # Calculate slope over the moving average
        if len(moving_avg) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(epochs, moving_avg)
            metrics['slope'] = slope
            metrics['r_squared'] = r_value**2
            metrics['p_value'] = p_value
        else:
            metrics['slope'] = np.nan
            metrics['r_squared'] = np.nan
            metrics['p_value'] = np.nan
    else:
        metrics['slope'] = np.nan
        metrics['r_squared'] = np.nan
        metrics['p_value'] = np.nan
    
    # Calculate loss change over recent epochs
    if len(loss_array) >= 50:
        recent_loss_change = np.abs(loss_array[-1] - loss_array[-50]) / loss_array[-50]
        metrics['recent_loss_change'] = recent_loss_change
    else:
        metrics['recent_loss_change'] = np.nan
    
    # Calculate variance of recent losses
    if len(loss_array) >= 100:
        recent_variance = np.var(loss_array[-100:])
        metrics['recent_variance'] = recent_variance
    else:
        metrics['recent_variance'] = np.nan
    
    # Find where loss stabilizes (convergence epoch)
    convergence_epoch = find_convergence_epoch(loss_array, min_epochs)
    metrics['convergence_epoch'] = convergence_epoch
    
    return metrics

def find_convergence_epoch(loss_array, min_epochs=500, tolerance=1e-4):
    """Find the epoch where loss appears to converge."""
    
    if len(loss_array) < min_epochs:
        return None
    
    # Look for convergence after min_epochs
    for i in range(min_epochs, len(loss_array) - 50):
        # Check if loss is relatively stable for the next 50 epochs
        recent_losses = loss_array[i:i+50]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # If coefficient of variation is small, consider it converged
        if loss_mean > 0 and loss_std / loss_mean < tolerance:
            return i
    
    return None

def determine_convergence_status(loss_array, metrics, convergence_threshold=1e-4, min_epochs=500):
    """Determine if training has converged based on various metrics."""
    
    # Check if we have enough data
    if len(loss_array) < min_epochs:
        return 'insufficient_data'
    
    # Check if loss has converged
    if metrics['convergence_epoch'] is not None:
        return 'converged'
    
    # Check slope-based convergence
    if not np.isnan(metrics['slope']):
        if abs(metrics['slope']) < convergence_threshold and metrics['r_squared'] > 0.1:
            return 'likely_converged'
    
    # Check recent loss change
    if not np.isnan(metrics['recent_loss_change']):
        if metrics['recent_loss_change'] < convergence_threshold:
            return 'likely_converged'
    
    # Check if still improving significantly
    if len(loss_array) >= 100:
        recent_improvement = (loss_array[-100] - loss_array[-1]) / loss_array[-100]
        if recent_improvement > 0.01:  # Still improving by more than 1%
            return 'still_training'
    
    return 'unclear'

def plot_convergence_analysis(results, save_dir="figures"):
    """Plot comprehensive convergence analysis."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: All loss curves
    ax1 = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for (name, result), color in zip(results.items(), colors):
        loss_array = result['loss_array']
        status = result['convergence_status']
        
        # Use different line styles based on convergence status
        if status == 'converged':
            linestyle = '-'
            alpha = 0.8
        elif status == 'likely_converged':
            linestyle = '--'
            alpha = 0.7
        elif status == 'still_training':
            linestyle = ':'
            alpha = 0.6
        else:
            linestyle = '-.'
            alpha = 0.5
        
        ax1.plot(loss_array, label=f"{name} ({status})", 
                color=color, linestyle=linestyle, alpha=alpha, linewidth=1.5)
        
        # Mark convergence point if available
        if result['convergence_metrics']['convergence_epoch'] is not None:
            conv_epoch = result['convergence_metrics']['convergence_epoch']
            ax1.axvline(x=conv_epoch, color=color, alpha=0.5, linestyle=':')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Final loss comparison
    ax2 = axes[0, 1]
    names = list(results.keys())
    final_losses = [results[name]['final_loss'] for name in names]
    statuses = [results[name]['convergence_status'] for name in names]
    
    # Color bars based on convergence status
    colors = []
    for status in statuses:
        if status == 'converged':
            colors.append('green')
        elif status == 'likely_converged':
            colors.append('orange')
        elif status == 'still_training':
            colors.append('red')
        else:
            colors.append('gray')
    
    bars = ax2.bar(range(len(names)), final_losses, color=colors, alpha=0.7)
    ax2.set_xlabel('Training Run')
    ax2.set_ylabel('Final Loss')
    ax2.set_title('Final Loss Comparison')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Convergence metrics
    ax3 = axes[1, 0]
    slopes = []
    r_squareds = []
    valid_names = []
    
    for name in names:
        slope = results[name]['convergence_metrics']['slope']
        r_squared = results[name]['convergence_metrics']['r_squared']
        
        if not np.isnan(slope) and not np.isnan(r_squared):
            slopes.append(slope)
            r_squareds.append(r_squared)
            valid_names.append(name)
    
    if valid_names:
        scatter = ax3.scatter(slopes, r_squareds, c=range(len(valid_names)), 
                             cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('Slope of Recent Loss')
        ax3.set_ylabel('R² of Linear Fit')
        ax3.set_title('Convergence Metrics')
        ax3.grid(True, alpha=0.3)
        
        # Add threshold lines
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero slope')
        ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='R² = 0.1')
        ax3.legend()
    
    # Plot 4: Loss range vs convergence status
    ax4 = axes[1, 1]
    loss_ranges = [results[name]['loss_range'] for name in names]
    
    # Color points based on convergence status
    colors = []
    for status in statuses:
        if status == 'converged':
            colors.append('green')
        elif status == 'likely_converged':
            colors.append('orange')
        elif status == 'still_training':
            colors.append('red')
        else:
            colors.append('gray')
    
    ax4.scatter(range(len(names)), loss_ranges, c=colors, s=100, alpha=0.7)
    ax4.set_xlabel('Training Run')
    ax4.set_ylabel('Loss Range (max - min)')
    ax4.set_title('Loss Range vs Convergence Status')
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    create_convergence_summary(results, save_dir)

def create_convergence_summary(results, save_dir):
    """Create a summary table of convergence analysis."""
    
    summary_data = []
    
    for name, result in results.items():
        metrics = result['convergence_metrics']
        
        summary_data.append({
            'Training Run': name,
            'Final Loss': f"{result['final_loss']:.6f}",
            'Loss Range': f"{result['loss_range']:.6f}",
            'Convergence Status': result['convergence_status'],
            'Convergence Epoch': metrics['convergence_epoch'] if metrics['convergence_epoch'] else 'N/A',
            'Slope': f"{metrics['slope']:.2e}" if not np.isnan(metrics['slope']) else 'N/A',
            'R²': f"{metrics['r_squared']:.3f}" if not np.isnan(metrics['r_squared']) else 'N/A',
            'Recent Loss Change': f"{metrics['recent_loss_change']:.2e}" if not np.isnan(metrics['recent_loss_change']) else 'N/A'
        })
    
    # Save summary to file
    with open(os.path.join(save_dir, 'convergence_summary.txt'), 'w') as f:
        f.write("Convergence Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for data in summary_data:
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nConvergence summary saved to {os.path.join(save_dir, 'convergence_summary.txt')}")

# Example usage
def example_usage():
    """Example of how to use the convergence analysis."""
    
    # Generate example loss arrays
    np.random.seed(42)
    
    # Converged training
    epochs = 1000
    loss1 = 1.0 * np.exp(-np.arange(epochs) / 200) + 0.1 + 0.01 * np.random.randn(epochs)
    
    # Still training
    loss2 = 1.0 * np.exp(-np.arange(epochs) / 500) + 0.2 + 0.02 * np.random.randn(epochs)
    
    # Oscillating (not converged)
    loss3 = 0.5 + 0.3 * np.exp(-np.arange(epochs) / 300) + 0.1 * np.sin(np.arange(epochs) / 50) + 0.05 * np.random.randn(epochs)
    
    # Very slow convergence
    loss4 = 2.0 * np.exp(-np.arange(epochs) / 800) + 0.05 + 0.005 * np.random.randn(epochs)
    
    loss_arrays = [loss1, loss2, loss3, loss4]
    names = ['Fast Convergence', 'Slow Training', 'Oscillating', 'Very Slow']
    
    # Analyze convergence
    results = analyze_convergence(loss_arrays, names)
    
    return results

if __name__ == "__main__":
    results = example_usage() 