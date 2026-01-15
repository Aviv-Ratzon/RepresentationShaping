import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations

# --- 1. Setup Data & Latents (2 Dimensions) ---
# We create a grid of (s, a) values to define the manifold
# s in [-1, 1], a in [-0.1, 0.1]
s_values = np.linspace(-0.5, 0.5, 11)       # 4 values
a_values = np.linspace(-0.1, 0.1, 3)   # 4 values

# Create the cartesian product (grid) of s and a
# This gives us 16 distinct latent states (4x4)
Z_grid = np.array(np.meshgrid(s_values, a_values)).T.reshape(-1, 2)
s = Z_grid[:, 0]
a = Z_grid[:, 1]

N_SAMPLES = len(s) # 16

# Generator phi(z) = [s^2, s^3, a^2, a^3]
# Note: X has 4 continuous dimensions
x_continuous = np.column_stack([s*2, s, a*2, a])

# Task h(z) = (s + a)^3 + (s + a)^2
y_task = (s + a)**3 + (s + a)**2

print(f"--- Data Stats ---")
print(f"Latent Samples: {N_SAMPLES}")
print(f"Latent s range: [{s.min()}, {s.max()}]")
print(f"Latent a range: [{a.min()}, {a.max()}]")
print(f"X shape: {x_continuous.shape}")


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
        X_bool_list.append(col_bits)
        
    # Concatenate all feature bits horizontally
    # Shape: (N_SAMPLES, n_cols * n_bits)
    X_bool = np.hstack(X_bool_list)
    return X_bool

X_bool = quantize_matrix(x_continuous, BITS_PER_FEATURE)
print(f'X bool shape: {X_bool.shape}')
m = X_bool.shape[1]

# Check for collisions
unique_rows = np.unique(X_bool, axis=0)
if len(unique_rows) < N_SAMPLES:
    print(f"\n[ERROR] Collision! {len(unique_rows)} unique vs {N_SAMPLES} samples.")
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
        row_bits = X_bool[i]
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
    # Effective Latent Dimension d:
    # We have 16 samples. If they are independent, d = log2(16) = 4.
    # However, s and a are continuous manifolds.
    # The degree bound should hold relative to the complexity of the task on the manifold.
    
    print(f"Task is cubic (degree 3). Found degree {final_degree}.")
    if final_degree <= 3:
        print("Success: Matches theoretical expectation (Degree <= 3).")
    else:
        print("Note: Degree is higher than 3. This is common with discretization noise.")
else:
    print("Failed to find solution in search range.")