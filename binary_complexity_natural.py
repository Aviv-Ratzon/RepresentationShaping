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

def FW_transform(x, y):
    n=x.shape[1]
    f_hat = np.zeros(2**n)
    S_l = [
    [i for i, bit in enumerate(mask) if bit]
    for mask in product([0, 1], repeat=n)
    ]

    for i, S in enumerate(S_l):
        xi_i = np.prod(x[:, S], axis=1)
        f_hat[i] = np.mean(xi_i*y)
    return f_hat

s_bits = 10
a_bits = 3
s_in = np.eye(s_bits)*2 - 1
a_in = np.eye(a_bits)*2 - 1
input_bits = 20
latent_bits = s_bits + a_bits

def generate_observations_in(latent_vec, n_features, latent_bits):
    observations = []
    for vec in latent_vec:
        # features = []
        # for _ in range(n_features):
        #     size = np.random.randint(latent_bits-2, latent_bits)  # 4, 5, or 6
        #     idx = np.random.choice(latent_bits, size=size, replace=False)
        #     product = 1
        #     for i in idx:
        #         product *= vec[i]
        #     features.append(product)
        # observations.append(features)
        observations.append([vec[i] if i==0 else vec[i]*vec[i+1] for i in range(len(vec)-1)])
    return observations
observations_in_s = generate_observations_in(s_in, s_bits, s_bits)
observations_in_a = generate_observations_in(a_in, a_bits, a_bits)

factor = 1
obs_size = s_bits * factor
obs_string = np.random.choice([-1, 1], size=obs_size+s_bits)
action_size = a_bits * factor
action_string = np.random.choice([-1, 1], size=action_size+a_bits)

out_latent = []
latent_bool = []
X_bool = []
for s in range(s_bits):
    for a_i, a in enumerate(np.arange(-(a_bits//2), a_bits//2+1)):
        if s + a < 0 or s + a >= s_bits:
            continue
        latent_bool.append(np.concatenate([s_in[s], a_in[a_i]], axis=0))
        # X_bool.append(np.concatenate([obs_string[s:s+obs_size], action_string[a_i:a_i+action_size]], axis=0))
        out_latent.append(s_in[s+a][-1])
        X_bool.append(np.concatenate([observations_in_s[s], observations_in_a[a_i]], axis=0))
# y_task = generate_twisted_polynomial_vector(np.array(out_latent)/s_bits, input_dim=1).squeeze()
y_task = np.array(out_latent)

X_bool = np.array(X_bool)
latent_bool = np.array(latent_bool)
y_task = np.array(y_task)

N_SAMPLES = X_bool.shape[0]
# y_task = generate_twisted_polynomial_vector(np.arange(s_bits), input_dim=1).squeeze()


print('Observation string: ')
print(obs_string)
print('Sample latent: ')
print(latent_bool[0])
print('Sample observations: ')
print(X_bool[0])
print('Sample task: ')
print(y_task[0])

# latent_bool = s_in
# X_bool = np.array([obs_string[i:i+obs_size] for i in range(s_bits)])

for data_bool, tag in zip([latent_bool, X_bool], ['latent', 'observations']):
    print(f'\n--------------- {tag} ---------------')
    print(f'X bool shape: {data_bool.shape}')
    m = data_bool.shape[1]

    # --- 3. Iterative Search for h* ---
    # We look for the lowest degree polynomial that solves the task.
    # We expect degree 3 because the task is cubic: (s+a)^3...

    max_search_degree = 10 # We search up to degree 4
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
        if final_degree <= latent_bool.shape[1]:
            print(f"Success: Matches theoretical expectation (Degree <= {latent_bool.shape[1]}).")
        else:
            print(f"Note: Degree is higher than {latent_bool.shape[1]}. This is common with discretization noise.")
    else:
        print("Failed to find solution in search range.")