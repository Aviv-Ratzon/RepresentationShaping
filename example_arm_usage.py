#!/usr/bin/env python3
"""
Simple example showing how to use the new arm data module.
"""

from run_sim import Config, run_sim
from data_modules import create_data

def example_basic_arm():
    """Basic example of using the arm data module."""
    
    print("Basic Arm Data Example")
    print("=" * 40)
    
    # Create configuration
    C = Config()
    C.data_geometry = 'arm'  # Use the new arm data module
    C.num_samples = 1000     # Generate 1000 samples
    C.print_progress = True  # Show progress information
    
    # Generate data
    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
    
    print(f"Generated {len(X)} samples")
    print(f"Input features: {input_size}")
    print(f"Output features: {output_size}")
    
    return X, y

def example_custom_arm():
    """Example with custom arm parameters."""
    
    print("\nCustom Arm Data Example")
    print("=" * 40)
    
    # Create configuration with custom parameters
    C = Config()
    C.data_geometry = 'arm'
    C.arm_length_1 = 2.0      # First segment length
    C.arm_length_2 = 1.5      # Second segment length
    C.angle_range = np.pi/4    # Smaller angle changes (45 degrees)
    C.noise_std = 0.02        # Add some noise
    C.num_samples = 500
    C.print_progress = True
    
    # Generate data
    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
    
    print(f"Generated {len(X)} samples with custom parameters")
    print(f"Arm lengths: {C.arm_length_1}, {C.arm_length_2}")
    print(f"Angle range: ±{C.angle_range:.2f} radians")
    print(f"Noise std: {C.noise_std}")
    
    return X, y

def example_with_training():
    """Example showing how to use arm data with the training framework."""
    
    print("\nArm Data with Training Example")
    print("=" * 40)
    
    # Create configuration for training
    C = Config()
    C.data_geometry = 'arm'
    C.num_samples = 2000
    C.arm_length_1 = 1.5
    C.arm_length_2 = 1.0
    C.angle_range = np.pi/3
    C.noise_std = 0.01
    
    # Training parameters
    C.hidden_size = 64
    C.L = 4
    C.learning_rate = 0.01
    C.num_epochs = 1000
    C.print_progress = True
    
    print("Running simulation with arm data...")
    print("(This may take a moment)")
    
    # Run the simulation
    results = run_sim(C)
    
    print("Training completed!")
    print(f"Final loss: {results['final_loss']:.6f}")
    
    return results

if __name__ == "__main__":
    import numpy as np
    
    # Run basic example
    X1, y1 = example_basic_arm()
    
    # Run custom example
    X2, y2 = example_custom_arm()
    
    # Run training example (commented out to avoid long execution)
    # results = example_with_training()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)
    print("\nTo run training with arm data, uncomment the training example.")
    print("To visualize the data, run: python test_arm_data.py")
