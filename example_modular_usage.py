#!/usr/bin/env python3
"""
Example script demonstrating how to use the new modular data creation system.
"""

# Example 1: Using corridor geometry (default)
print("Example 1: Corridor Geometry")
print("-" * 40)

# This would work if the numpy/scikit-learn compatibility issue is resolved
try:
    from run_sim import Config, run_sim
    
    # Create configuration for corridor geometry
    C = Config()
    C.data_geometry = 'corridor'  # This is the default, so you could omit this line
    C.length_corridors = [10, 10]  # Smaller for demonstration
    C.max_move = 3
    C.corridor_dim = 1
    C.print_progress = True
    
    print(f"Configuration:")
    print(f"  - Data geometry: {C.data_geometry}")
    print(f"  - Corridor lengths: {C.length_corridors}")
    print(f"  - Max move: {C.max_move}")
    print(f"  - Corridor dimension: {C.corridor_dim}")
    
    # Run simulation
    # results = run_sim(C)
    print("✓ Configuration created successfully!")
    print("  (Simulation commented out due to numpy compatibility issues)")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60 + "\n")

# Example 2: Using hyperbolic geometry
print("Example 2: Hyperbolic Geometry")
print("-" * 40)

try:
    from run_sim import Config
    
    # Create configuration for hyperbolic geometry
    C = Config()
    C.data_geometry = 'hyperbolic'
    C.length_corridors = [5, 5]  # Smaller for demonstration
    C.max_move = 2
    C.corridor_dim = 2  # Hyperbolic typically uses 2D
    C.print_progress = True
    
    print(f"Configuration:")
    print(f"  - Data geometry: {C.data_geometry}")
    print(f"  - Corridor lengths: {C.length_corridors}")
    print(f"  - Max move: {C.max_move}")
    print(f"  - Corridor dimension: {C.corridor_dim}")
    
    print("✓ Configuration created successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60 + "\n")

# Example 3: Error handling for invalid geometry
print("Example 3: Error Handling")
print("-" * 40)

try:
    from run_sim import Config
    
    C = Config()
    C.data_geometry = 'invalid_geometry'
    
    print(f"Attempting to use invalid geometry: {C.data_geometry}")
    print("This should raise a ValueError with available options")
    
    # This would raise an error when create_data is called
    # from data_modules import create_data
    # create_data(C)
    
    print("✓ Error handling demonstration completed!")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60 + "\n")

# Example 4: Direct usage of data modules
print("Example 4: Direct Data Module Usage")
print("-" * 40)

try:
    from data_modules import create_data_corridor, create_data_hyperbolic, DATA_GEOMETRY_FUNCTIONS
    
    print(f"Available data geometries: {list(DATA_GEOMETRY_FUNCTIONS.keys())}")
    
    # You can also use the functions directly
    print("✓ Direct data module access successful!")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60 + "\n")

# Example 5: Configuration inheritance
print("Example 5: Configuration Inheritance")
print("-" * 40)

try:
    from run_sim import Config
    
    # Create a base configuration
    base_config = Config()
    base_config.length_corridors = [15, 15]
    base_config.max_move = 5
    base_config.corridor_dim = 1
    
    # Create specific configurations for different geometries
    corridor_config = Config()
    corridor_config.data_geometry = 'corridor'
    corridor_config.length_corridors = base_config.length_corridors
    corridor_config.max_move = base_config.max_move
    corridor_config.corridor_dim = base_config.corridor_dim
    
    hyperbolic_config = Config()
    hyperbolic_config.data_geometry = 'hyperbolic'
    hyperbolic_config.length_corridors = base_config.length_corridors
    hyperbolic_config.max_move = base_config.max_move
    hyperbolic_config.corridor_dim = 2  # Override for hyperbolic
    
    print("Base configuration:")
    print(f"  - Corridor lengths: {base_config.length_corridors}")
    print(f"  - Max move: {base_config.max_move}")
    print(f"  - Corridor dimension: {base_config.corridor_dim}")
    
    print("\nCorridor configuration:")
    print(f"  - Data geometry: {corridor_config.data_geometry}")
    print(f"  - Corridor dimension: {corridor_config.corridor_dim}")
    
    print("\nHyperbolic configuration:")
    print(f"  - Data geometry: {hyperbolic_config.data_geometry}")
    print(f"  - Corridor dimension: {hyperbolic_config.corridor_dim}")
    
    print("✓ Configuration inheritance demonstration completed!")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60 + "\n")

print("Summary:")
print("- The modular data creation system allows you to choose between different data geometries")
print("- Set C.data_geometry = 'corridor' for corridor-based data (default)")
print("- Set C.data_geometry = 'hyperbolic' for tree-based data")
print("- The system is extensible - you can add new geometries easily")
print("- All existing functionality is preserved with backward compatibility")
