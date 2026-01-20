import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander


def generate_twisted_polynomial_vector(x, input_dim=100, min_deg=5, max_deg=10):
    """
    Generates a high-dimensional vector based on x using high-degree polynomials.
    
    Args:
        x (float or np.array): Scalar input in range [-1, 1].
        input_dim (int): The dimension of the output vector.
        min_deg (int): The lowest degree polynomial to use (prevents linearity).
        max_deg (int): The highest degree polynomial to use.
        
    Returns:
        np.array: A high-dimensional vector (or matrix if x is an array).
    """
    # 1. Generate the Vandermonde matrix for Chebyshev polynomials
    # This calculates T_0(x), T_1(x), ... T_max_deg(x)
    # Shape: (n_samples, max_deg + 1)
    polys = chebvander(x, max_deg)
    
    # 2. Slice to keep only high frequencies
    # We discard degrees 0 to min_deg-1 to ensure no monotonic trends exist
    high_freq_polys = polys[:, min_deg:max_deg+1]
    
    # 3. Create a random mixing matrix (fixed seed for reproducibility)
    # We project the polynomial basis into the desired high dimension
    rng = np.random.RandomState(42)
    feature_dim = high_freq_polys.shape[1]
    W = rng.randn(feature_dim, input_dim)
    
    # 4. Mix the polynomials
    high_dim_vector = high_freq_polys @ W
    
    return high_dim_vector

s_bits = 10
s_in = np.eye(s_bits)*2 - 1

N_SAMPLES = s_bits

# y_task = np.arange(s_bits)
y_task = generate_twisted_polynomial_vector(np.arange(s_bits), input_dim=1).squeeze()

obs_size = 40
obs_string = np.random.choice([-1, 1], size=obs_size+s_bits)

print('Observed string: ')
print(obs_string)

latent_bool = s_in
X_bool = np.array([obs_string[i:i+obs_size] for i in range(s_bits)])

for data_bool, tag in zip([latent_bool, X_bool], ['latent', 'observations']):
    print(f'\n--------------- {tag} ---------------')
    print(f'X bool shape: {data_bool.shape}')
    m = data_bool.shape[1]

    # --- 3. Iterative Search for h* ---
    # We look for the lowest degree polynomial that solves the task.
    # We expect degree 3 because the task is cubic: (s+a)^3...

    max_search_degree = 4 # We search up to degree 4
    all_indices = range(m)

    # Pre-generate basis functions is too slow for m=20 if we do ALL combinations.
    # Optimization: We generate basis functions on the fly or limit the search space.
    # For this demo, we use the same strict logic but optimized loop.

    print("\n--- Solving for h* ---")

    final_degree = -1

    for d_try in range(max_search_degree + 1):
        print(f"Checking Degree {d_try}...", end="")
        
        # Generate all subsets of size 'd_try'
        # For m=20, d=3 is manageable (1140 combinations), d=4 is 4845.
        basis_subsets = list(combinations(all_indices, d_try))
        
        # Build Design Matrix for THIS degree ONLY (to check if we can add it to previous)
        # Actually, proper forward selection adds degrees cumulatively.
        # Let's rebuild the FULL matrix up to degree d_try.
        
        full_basis_indices = []
        for r in range(d_try + 1):
            full_basis_indices.extend(list(combinations(all_indices, r)))
            
        n_features = len(full_basis_indices)
        print(f" (Features: {n_features}) -> ", end="")
        
        # Construct A matrix
        A = np.zeros((N_SAMPLES, n_features))
        
        for i in range(N_SAMPLES):
            # We need to compute product of bits for each subset
            # Optimization: Use fancy indexing if possible, or simple loop
            # Given small N_SAMPLES, loop is fine.
            row_bits = data_bool[i]
            for col, subset in enumerate(full_basis_indices):
                val = 1
                for idx in subset:
                    val *= row_bits[idx]
                A[i, col] = val
                
        # Solve
        model = LinearRegression(fit_intercept=False)
        model.fit(A, y_task)
        preds = model.predict(A)
        mse = np.mean((preds - y_task)**2)
        
        print(f"MSE: {mse:.10f}")
        
        if mse < 1e-9:
            final_degree = d_try
            break

    # --- 4. Analysis ---
    print("\n--- Results ---")
    if final_degree != -1:
        print(f"Found h* at Degree: {final_degree}")
        if final_degree <= 3:
            print("Success: Matches theoretical expectation (Degree <= 3).")
        else:
            print("Note: Degree is higher than 3. This is common with discretization noise.")
    else:
        print("Failed to find solution in search range.")