# Simple approach: Import all functions from utils for easy access
from utils import *

# Create a list of all available functions
import inspect
import utils

# Get all function names
all_functions = [name for name, obj in inspect.getmembers(utils) 
                 if inspect.isfunction(obj) and obj.__module__ == 'utils']

# Print available functions
def show_functions():
    """Show all available functions from utils"""
    print("Available functions from utils:")
    print("=" * 40)
    for i, func_name in enumerate(sorted(all_functions), 1):
        print(f"{i:2d}. {func_name}")
    print(f"\nTotal: {len(all_functions)} functions")

# Quick search function
def find_function(keyword):
    """Find functions containing the keyword"""
    matches = [name for name in all_functions if keyword.lower() in name.lower()]
    if matches:
        print(f"Functions containing '{keyword}':")
        for match in matches:
            print(f"  {match}")
    else:
        print(f"No functions found containing '{keyword}'")
    return matches

# Example usage
if __name__ == "__main__":
    show_functions()
    print("\nExample searches:")
    find_function("matrix")
    find_function("cosine")
    find_function("gradient") 