import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations
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
# --- 1. Setup Data & Latents (2 Dimensions) ---
# We create a grid of (s, a) values to define the manifold
# s in [-1, 1], a in [-0.1, 0.1]
s_bits = 4
a_bits = 4
s_values = np.linspace(-1, 1, 2**s_bits)       # 4 values
a_values = np.linspace(-1, 1, 2**a_bits)   # 4 values

# Create the cartesian product (grid) of s and a
# This gives us 16 distinct latent states (4x4)
Z_grid = np.array(np.meshgrid(s_values, a_values)).T.reshape(-1, 2)
s = Z_grid[:, [0]]
a = Z_grid[:, [1]]

N_SAMPLES = len(s) # 16

# Generator phi(z) = [s^2, s^3, a^2, a^3]
# Note: X has 4 continuous dimensions
latent_continuous = np.concatenate([s, a], axis=1)
proj = np.random.randn(1,5)+1
# x_continuous = np.concatenate([(s@proj), (a@proj)], axis=1)
x_continuous = np.concatenate([generate_twisted_polynomial_vector(s.squeeze(), input_dim=5),
                               generate_twisted_polynomial_vector(a.squeeze(), input_dim=5)  ], axis=1)


identical_inputs_inds = np.where((abs(x_continuous[None,:, :] - x_continuous[:, None, :]).sum(-1)) < 1e-9)
for i, j in zip(identical_inputs_inds[0], identical_inputs_inds[1]):
    if i != j:
        print(f"Identical input at {i}, {j}: {x_continuous[i]} == {x_continuous[j]}, where s={s[i]} and a={a[i]} == s={s[j]} and a={a[j]}")
# Task h(z) = (s + a)^3 + (s + a)^2
y_task = (s + a)**3 + (s + a)**2

print(f"--- Data Stats ---")
print(f"Latent Samples: {N_SAMPLES}")
print(f"Latent s range: [{s.min()}, {s.max()}]")
print(f"Latent a range: [{a.min()}, {a.max()}]")
print(f"X shape: {x_continuous.shape}")

def check_binary_encoding(X_vals, X_bool):
    bits_mat = np.array(X_bool)
    bits_dist = abs(bits_mat[None, :, :] - bits_mat[:, None, :]).sum(-1)
    val_dist = abs(X_vals[None, :] - X_vals[:, None])
    mismatch = np.where(((bits_dist == 0) & (val_dist > 0)) | ((bits_dist > 0) & (val_dist == 0)))
    for i, j in zip(mismatch[0], mismatch[1]):
        print(f"Mismatch at {i}, {j}: {X_vals[i]} == {X_vals[j]}")
    return mismatch
    

# --- 2. Robust Boolean Encoding ---
# We need to encode 4 continuous features into bits.
# Crucial: We must normalize EACH column independently because 
# a^2 (approx 0.01) is much smaller than s^2 (approx 1.0).

BITS_PER_FEATURE = 10 # Total bits m = 4 features * 5 bits = 20 bits

def quantize_matrix(X, n_bits):
    """Quantizes each column of X to n_bits independently."""
    n_rows, n_cols = X.shape
    X_bool_list = []
    
    for col_idx in range(n_cols):
        col_vals = X[:, col_idx]
        col_vals = np.round(col_vals, 6)
        min_v, max_v = col_vals.min(), col_vals.max()
        
        # Avoid division by zero if a feature is constant
        if max_v == min_v:
            max_v += 1e-9
            
        # Normalize to 0..1
        norm = (col_vals - min_v) / (max_v - min_v)
        
        # Scale to integer levels
        levels = 2**n_bits
        ints = np.floor(norm * levels).astype(int)
        ints = np.clip(ints, 0, levels - 1)
        
        # Convert integers to bit vectors [-1, 1]
        col_bits = []
        for val in ints:
            bits = [1 if (val >> b) & 1 else -1 for b in range(n_bits)]
            col_bits.append(bits)
        check_binary_encoding(col_vals, col_bits)
        X_bool_list.append(col_bits)
        
    # Concatenate all feature bits horizontally
    # Shape: (N_SAMPLES, n_cols * n_bits)
    X_bool = np.hstack(X_bool_list)
    return X_bool

s_bool = quantize_matrix(s, s_bits)
a_bool = quantize_matrix(a, a_bits)
latent_bool = np.concatenate([s_bool, a_bool], axis=1)
X_bool = quantize_matrix(x_continuous, BITS_PER_FEATURE)

for data_bool, data_cont, tag in zip([latent_bool, X_bool], [latent_continuous, x_continuous], ['latent', 'observations']):
    print(f'\n--------------- {tag} ---------------')
    print(f'X bool shape: {data_bool.shape}')
    m = data_bool.shape[1]

    # Check for collisions
    unique_rows = np.unique(data_bool, axis=0)
    unique_samples = np.unique(data_cont, axis=0)
    if len(unique_rows) < len(unique_samples):
        print(f"\n[ERROR] Collision! {len(unique_rows)} unique bool vs {len(unique_samples)} unique samples.")
        exit()
    else:
        print(f"\n[OK] Encoding clean. Input Dimension m = {m} bits")


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