#!/usr/bin/env python3
"""
Test script for the new arm data module.
Demonstrates how to create and visualize the robotic arm dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from run_sim import Config
from data_modules import create_data_arm

def test_arm_data():
    """Test the arm data generation with different configurations."""
    
    print("Testing Arm Data Module")
    print("=" * 50)
    
    # Test 1: Basic arm data
    print("\n1. Basic Arm Data (100 samples)")
    C1 = Config()
    C1.data_geometry = 'arm'
    C1.num_samples = 100
    C1.print_progress = True
    
    X1, y1, corridor1, loc_X1, loc_y1, action_taken1, dim_l1, input_size1, output_size1, n_actions1 = create_data_arm(C1)
    
    print(f"Input shape: {X1.shape}")
    print(f"Output shape: {y1.shape}")
    print(f"Sample input: {X1[0]}")
    print(f"Sample output: {y1[0]}")
    
    # Test 2: Custom arm lengths
    print("\n2. Custom Arm Lengths")
    C2 = Config()
    C2.data_geometry = 'arm'
    C2.arm_length_1 = 2.0  # Longer first segment
    C2.arm_length_2 = 1.5  # Medium second segment
    C2.num_samples = 50
    C2.print_progress = True
    
    X2, y2, corridor2, loc_X2, loc_y2, action_taken2, dim_l2, input_size2, output_size2, n_actions2 = create_data_arm(C2)
    
    # Test 3: Different angle range
    print("\n3. Smaller Angle Range")
    C3 = Config()
    C3.data_geometry = 'arm'
    C3.angle_range = np.pi / 4  # Smaller angle changes (45 degrees)
    C3.num_samples = 50
    C3.print_progress = True
    
    X3, y3, corridor3, loc_X3, loc_y3, action_taken3, dim_l3, input_size3, output_size3, n_actions3 = create_data_arm(C3)
    
    # Test 4: With noise
    print("\n4. With Noise")
    C4 = Config()
    C4.data_geometry = 'arm'
    C4.noise_std = 0.05  # Add some noise
    C4.num_samples = 50
    C4.print_progress = True
    
    X4, y4, corridor4, loc_X4, loc_y4, action_taken4, dim_l4, input_size4, output_size4, n_actions4 = create_data_arm(C4)
    
    return X1, y1, X2, y2, X3, y3, X4, y4

def visualize_arm_data(X, y, title="Arm Data Visualization"):
    """Visualize the arm data by showing initial and final positions."""
    
    # Extract positions from input and output
    elbow_x_init = X[:, 0]
    elbow_y_init = X[:, 1]
    wrist_x_init = X[:, 2]
    wrist_y_init = X[:, 3]
    
    elbow_x_final = y[:, 0]
    elbow_y_final = y[:, 1]
    wrist_x_final = y[:, 2]
    wrist_y_final = y[:, 3]
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot initial positions
    plt.subplot(2, 2, 1)
    plt.scatter(elbow_x_init, elbow_y_init, alpha=0.6, label='Elbow (initial)', s=20)
    plt.scatter(wrist_x_init, wrist_y_init, alpha=0.6, label='Wrist (initial)', s=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Initial Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot final positions
    plt.subplot(2, 2, 2)
    plt.scatter(elbow_x_final, elbow_y_final, alpha=0.6, label='Elbow (final)', s=20)
    plt.scatter(wrist_x_final, wrist_y_final, alpha=0.6, label='Wrist (final)', s=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Final Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot movement vectors
    plt.subplot(2, 2, 3)
    for i in range(min(20, len(X))):  # Show first 20 samples
        # Elbow movement
        plt.arrow(elbow_x_init[i], elbow_y_init[i], 
                 elbow_x_final[i] - elbow_x_init[i], 
                 elbow_y_final[i] - elbow_y_init[i],
                 head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.6)
        # Wrist movement
        plt.arrow(wrist_x_init[i], wrist_y_init[i], 
                 wrist_x_final[i] - wrist_x_init[i], 
                 wrist_y_final[i] - wrist_y_init[i],
                 head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Movement Vectors (First 20 samples)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot angle changes
    plt.subplot(2, 2, 4)
    delta_theta = X[:, 4]  # Shoulder angle change
    delta_phi = X[:, 5]    # Elbow angle change
    plt.scatter(delta_theta, delta_phi, alpha=0.6)
    plt.xlabel('Delta Theta (Shoulder)')
    plt.ylabel('Delta Phi (Elbow)')
    plt.title('Angle Changes')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()

def test_with_run_sim():
    """Test the arm data module with the run_sim framework."""
    
    print("\n" + "=" * 50)
    print("Testing with run_sim framework")
    print("=" * 50)
    
    # Create configuration for arm data
    C = Config()
    C.data_geometry = 'arm'
    C.num_samples = 500
    C.arm_length_1 = 1.5
    C.arm_length_2 = 1.0
    C.angle_range = np.pi / 3  # 60 degrees
    C.noise_std = 0.02
    C.print_progress = True
    
    # Generate data using the modular system
    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
    
    print(f"\nData generated successfully!")
    print(f"Input features: {input_size}")
    print(f"Output features: {output_size}")
    print(f"Number of samples: {len(X)}")
    
    # Show a few examples
    print(f"\nFirst 3 samples:")
    for i in range(3):
        print(f"Sample {i+1}:")
        print(f"  Input:  Elbow({X[i,0]:.3f}, {X[i,1]:.3f}), Wrist({X[i,2]:.3f}, {X[i,3]:.3f}), Δθ={X[i,4]:.3f}, Δφ={X[i,5]:.3f}")
        print(f"  Output: Elbow({y[i,0]:.3f}, {y[i,1]:.3f}), Wrist({y[i,2]:.3f}, {y[i,3]:.3f})")
        print()
    
    return X, y

if __name__ == "__main__":
    # Test the arm data module directly
    X1, y1, X2, y2, X3, y3, X4, y4 = test_arm_data()
    
    # Visualize the first dataset
    visualize_arm_data(X1, y1, "Basic Arm Data (100 samples)")
    
    # Test with run_sim framework
    from data_modules import create_data
    X, y = test_with_run_sim()
    
    # Visualize the run_sim dataset
    visualize_arm_data(X, y, "Arm Data via run_sim Framework")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)
