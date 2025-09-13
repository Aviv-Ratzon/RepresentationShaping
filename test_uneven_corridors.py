"""
Test script to verify the new create_data_uneven_corridors function works correctly.
"""

import numpy as np
import sys
import os

# Add the current directory to the path so we can import data_modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_modules import create_data_uneven_corridors

class Config:
    """Simple configuration class for testing."""
    def __init__(self):
        # Basic parameters
        self.corridor_widths = [3, 4]  # Widths for the two corridors
        self.corridor_lengths = [5, 6]  # Lengths for the two corridors
        self.max_move = 2
        self.min_move = 0
        self.print_progress = True
        
        # Action parameters
        self.split_actions = False
        self.allow_backwards = True
        self.one_hot_actions = True
        
        # Input parameters
        self.one_hot_inputs = True
        self.input_size = 10
        self.input_smoothing = 0.1
        
        # Movement parameters
        self.egocentric_movement = False
        self.cyclic_corridors = False
        self.mask_states = None
        
        # Data processing
        self.whiten_data = False

def test_basic_functionality():
    """Test basic functionality of the uneven corridors module."""
    print("=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    C = Config()
    
    try:
        X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data_uneven_corridors(C)
        
        print(f"✓ Basic functionality test passed!")
        print(f"  - Input shape: {X.shape}")
        print(f"  - Output shape: {y.shape}")
        print(f"  - Input size: {input_size}")
        print(f"  - Output size: {output_size}")
        print(f"  - Number of actions: {n_actions}")
        print(f"  - Corridor distribution: {np.bincount(corridor)}")
        print(f"  - Dimension distribution: {np.bincount(dim_l)}")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_split_actions():
    """Test with split actions enabled."""
    print("\n" + "=" * 60)
    print("TESTING SPLIT ACTIONS")
    print("=" * 60)
    
    C = Config()
    C.split_actions = True
    
    try:
        X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data_uneven_corridors(C)
        
        print(f"✓ Split actions test passed!")
        print(f"  - Input shape: {X.shape}")
        print(f"  - Output shape: {y.shape}")
        print(f"  - Number of actions: {n_actions}")
        print(f"  - Corridor distribution: {np.bincount(corridor)}")
        
        return True
    except Exception as e:
        print(f"✗ Split actions test failed: {e}")
        return False

def test_different_sizes():
    """Test with different corridor sizes."""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT CORRIDOR SIZES")
    print("=" * 60)
    
    C = Config()
    C.corridor_widths = [2, 8]
    C.corridor_lengths = [3, 4]
    
    try:
        X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data_uneven_corridors(C)
        
        print(f"✓ Different sizes test passed!")
        print(f"  - Input shape: {X.shape}")
        print(f"  - Output shape: {y.shape}")
        print(f"  - Corridor 1 states: {C.corridor_widths[0] * C.corridor_lengths[0]}")
        print(f"  - Corridor 2 states: {C.corridor_widths[1] * C.corridor_lengths[1]}")
        print(f"  - Total states: {C.corridor_widths[0] * C.corridor_lengths[0] + C.corridor_widths[1] * C.corridor_lengths[1]}")
        print(f"  - Corridor distribution: {np.bincount(corridor)}")
        
        return True
    except Exception as e:
        print(f"✗ Different sizes test failed: {e}")
        return False

def test_cyclic_corridors():
    """Test with cyclic corridors enabled."""
    print("\n" + "=" * 60)
    print("TESTING CYCLIC CORRIDORS")
    print("=" * 60)
    
    C = Config()
    C.cyclic_corridors = True
    
    try:
        X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data_uneven_corridors(C)
        
        print(f"✓ Cyclic corridors test passed!")
        print(f"  - Input shape: {X.shape}")
        print(f"  - Output shape: {y.shape}")
        print(f"  - Number of actions: {n_actions}")
        print(f"  - Corridor distribution: {np.bincount(corridor)}")
        
        return True
    except Exception as e:
        print(f"✗ Cyclic corridors test failed: {e}")
        return False

def test_non_one_hot():
    """Test with non-one-hot inputs."""
    print("\n" + "=" * 60)
    print("TESTING NON-ONE-HOT INPUTS")
    print("=" * 60)
    
    C = Config()
    C.one_hot_inputs = False
    C.input_size = 5
    
    try:
        X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data_uneven_corridors(C)
        
        print(f"✓ Non-one-hot inputs test passed!")
        print(f"  - Input shape: {X.shape}")
        print(f"  - Output shape: {y.shape}")
        print(f"  - Input size: {input_size}")
        print(f"  - Output size: {output_size}")
        print(f"  - Number of actions: {n_actions}")
        
        return True
    except Exception as e:
        print(f"✗ Non-one-hot inputs test failed: {e}")
        return False

def test_data_consistency():
    """Test that the data is consistent."""
    print("\n" + "=" * 60)
    print("TESTING DATA CONSISTENCY")
    print("=" * 60)
    
    C = Config()
    C.corridor_widths = [2, 3]
    C.corridor_lengths = [2, 3]
    C.max_move = 1
    
    try:
        X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data_uneven_corridors(C)
        
        # Check that all locations are within bounds
        for i, (cor, loc, next_loc) in enumerate(zip(corridor, loc_X, loc_y)):
            width = C.corridor_widths[cor]
            length = C.corridor_lengths[cor]
            
            # Check current location
            if not (0 <= loc[0] < width and 0 <= loc[1] < length):
                print(f"✗ Invalid current location {loc} for corridor {cor} (size {width}x{length})")
                return False
            
            # Check next location
            if not (0 <= next_loc[0] < width and 0 <= next_loc[1] < length):
                print(f"✗ Invalid next location {next_loc} for corridor {cor} (size {width}x{length})")
                return False
        
        print(f"✓ Data consistency test passed!")
        print(f"  - All locations are within bounds")
        print(f"  - Total samples: {len(X)}")
        print(f"  - Corridor distribution: {np.bincount(corridor)}")
        print(f"  - Dimension distribution: {np.bincount(dim_l)}")
        
        return True
    except Exception as e:
        print(f"✗ Data consistency test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("TESTING UNEVEN CORRIDORS DATA MODULE")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_split_actions,
        test_different_sizes,
        test_cyclic_corridors,
        test_non_one_hot,
        test_data_consistency
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The uneven corridors module is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
