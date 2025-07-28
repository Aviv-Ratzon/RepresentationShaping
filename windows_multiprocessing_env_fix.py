import os
import sys

def setup_windows_multiprocessing():
    """Set environment variables to fix Windows multiprocessing issues"""
    
    # Set multiprocessing start method to spawn (Windows default, but explicit)
    os.environ['PYTHONPATH'] = os.getcwd()
    
    # Increase the default timeout for multiprocessing operations
    os.environ['JOBLIB_TIMEOUT'] = '300'  # 5 minutes
    
    # Set joblib backend to use fewer workers
    os.environ['JOBLIB_MAX_NBYTES'] = '100M'  # Limit memory per worker
    
    # Force joblib to use loky backend which is more stable on Windows
    os.environ['JOBLIB_BACKEND'] = 'loky'
    
    print("Windows multiprocessing environment configured:")
    print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"  JOBLIB_TIMEOUT: {os.environ.get('JOBLIB_TIMEOUT', 'Not set')}")
    print(f"  JOBLIB_MAX_NBYTES: {os.environ.get('JOBLIB_MAX_NBYTES', 'Not set')}")
    print(f"  JOBLIB_BACKEND: {os.environ.get('JOBLIB_BACKEND', 'Not set')}")

def run_with_fixed_environment():
    """Run the analysis with proper Windows multiprocessing settings"""
    setup_windows_multiprocessing()
    
    # Import and run the analysis
    from analyze_margins_fixed import run_parallel_simulations
    
    try:
        results = run_parallel_simulations()
        print("Analysis completed successfully!")
        return results
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Falling back to sequential processing...")
        
        # Fallback to sequential processing
        from analyze_margins_sequential import run_sequential_simulations
        return run_sequential_simulations()

if __name__ == '__main__':
    results = run_with_fixed_environment() 