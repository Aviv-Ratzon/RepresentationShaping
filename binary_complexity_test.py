import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from itertools import combinations, product
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander


def get_all_basis_functions(n):
    S_l = [
    [i for i, bit in enumerate(mask) if bit]
    for mask in product([0, 1], repeat=n)
    ]
    all_samples = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n)
    xi_l = []

    for S in S_l:
        xi_i = np.prod(all_samples[:, S], axis=1)
        xi_l.append(xi_i)
    return xi_l

def get_obs_fns(obs_bits, latent_bits):
    fn_l = []
    for _ in range(obs_bits):
        fn = lambda x: np.prod(x[np.random.choice(latent_bits, size=np.random.choice(range(min(2,latent_bits), latent_bits+1)), replace=False)])
        fn_l.append(fn)
    return fn_l

def generate_obs(latent_bool, obs_bits=None):
    if obs_bits is None:
        obs_bits = latent_bool.shape[1]
    obs_fns = get_obs_fns(obs_bits, latent_bool.shape[1])
    obs = [[fn(x) for fn in obs_fns] for x in latent_bool]
    return obs

def generate_latent(latent_bits, offset=0):
    latent_bools = np.array(np.meshgrid(*[[-1, 1]] * latent_bits)).T.reshape(-1, latent_bits)
    latent_vals = [(2**np.where(np.array(latent_bool)==-1)[0]).sum() for latent_bool in latent_bools]
    sorted_indices = np.argsort(latent_vals)
    latent_bools = latent_bools[sorted_indices]
    latent_vals = [latent_vals[i] + offset for i in sorted_indices]
    return latent_bools, latent_vals

# def generate_latent(latent_bits, offset=0):
#     latent_bools = np.array([np.concatenate([np.ones(i), -np.ones(latent_bits-i)]) for i in range(latent_bits)])
#     latent_vals = np.arange(latent_bits) + offset
#     return latent_bools, latent_vals

s_bits = 10
state_bools, state_vals = generate_latent(s_bits)

a_bits = 2
action_bools, action_vals = generate_latent(a_bits, offset=-2**(a_bits)//2)

s_obs_size = s_bits*3
a_obs_size = a_bits*3
state_obs_bools = generate_obs(state_bools, s_obs_size)
action_obs_bools = generate_obs(action_bools, a_obs_size)

# state_obs_bools = [[s_bool[i] if i in [0,1] else s_bool[i]*s_bool[i-1]*s_bool[i-2] for i in range(len(s_bool))] for s_bool in state_bools]
# action_obs_bools = [[a_bool[i] if i in [0,1] else a_bool[i]*a_bool[i-1]*a_bool[i-2] for i in range(len(a_bool))] for a_bool in action_bools]
# state_obs_bools = get_all_basis_functions(s_bits)
# action_obs_bools = get_all_basis_functions(a_bits)

X_bool = []
latent_bool = []
y_task = []
for s_bool, s_val in zip(state_bools, state_vals):
    for a_bool, a_val in zip(action_bools, action_vals):
        if s_val + a_val < 0 or s_val + a_val >= max(state_vals):
            continue

        X_bool.append(np.concatenate([state_obs_bools[s_val], action_obs_bools[a_val]], axis=0))
        latent_bool.append(np.array(state_bools[s_val+a_val]))
        # latent_bool.append(np.concatenate([s_bool, a_bool], axis=0))
        y_task.append(state_obs_bools[s_val+a_val][0])
        # y_task.append(s_val+a_val)

X_bool = np.array(X_bool)
latent_bool = np.array(latent_bool)
y_task = np.array(y_task)
N_SAMPLES = X_bool.shape[0]
# y_task = generate_twisted_polynomial_vector(np.arange(s_bits), input_dim=1).squeeze()


# latent_bool = s_in
# X_bool = np.array([obs_string[i:i+obs_size] for i in range(s_bits)])
degrees = []
for data_bool, tag in zip([latent_bool, X_bool], ['latent', 'observations']):
    print(f'\n--------------- {tag} ---------------')
    print(f'X bool shape: {data_bool.shape}')
    m = data_bool.shape[1]

    # --- 3. Iterative Search for h* ---
    # We look for the lowest degree polynomial that solves the task.
    # We expect degree 3 because the task is cubic: (s+a)^3...

    max_search_degree = data_bool.shape[1] # We search up to degree 4
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
                
        import torch

        device = "cuda"
        X = torch.as_tensor(A, device=device, dtype=torch.float32)
        y = torch.as_tensor(y_task, device=device, dtype=torch.float32).unsqueeze(1)

        # Add intercept column if you want a bias term
        X1 = torch.cat([torch.ones((X.shape[0], 1), device=device), X], dim=1)

        sol = torch.linalg.lstsq(X1, y).solution  # (n_features+1, 1)
        b0 = sol[0].item()
        w  = sol[1:].squeeze(1)
        preds = (X1 @ w).cpu().numpy()

        # model = LinearRegression(fit_intercept=False)
        # model.fit(A, y_task)
            
        # preds = model.predict(A)
        mse = np.mean((preds - y_task)**2)
        
        print(f"MSE: {mse:.10f}")
        
        if mse < 1e-9:
            # print('Coefficients: ', model.coef_)
            final_degree = d_try
            break

    # --- 4. Analysis ---
    print("\n--- Results ---")
    degrees.append(final_degree)
    if final_degree != -1:
        print(f"Found h* at Degree: {final_degree}")
        if final_degree <= latent_bool.shape[1]:
            print(f"Success: Matches theoretical expectation (Degree <= {latent_bool.shape[1]}).")
        else:
            print(f"Note: Degree is higher than {latent_bool.shape[1]}. This is common with discretization noise.")
    else:
        print("Failed to find solution in search range.")

print('-----------------------------')
print(f'Found {degrees[0]} for latent, {degrees[1]} for observations')