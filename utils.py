from itertools import product
from functools import reduce
import os
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import svds
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
from run_sim import run_sim_wrapper


def state_dict_to_theta(model_dict):
    """
    Convert a state dictionary to a flattened parameter vector theta.
    
    Args:
        model_dict: Dictionary containing model parameters
        
    Returns:
        theta: Flattened parameter vector
        shapes: List of original tensor shapes for reconstruction
        sizes: List of tensor sizes for reconstruction
    """
    W_l = [W.clone().detach() for W in model_dict.values()]
    shapes = [W.shape for W in W_l]
    sizes = [W.numel() for W in W_l]
    theta = torch.concatenate([W.reshape(-1) for W in model_dict.values()])
    return theta, shapes, sizes


def theta_to_state_dict(theta, model_dict, shapes=None, sizes=None):
    """
    Convert a flattened parameter vector theta back to a state dictionary.
    
    Args:
        theta: Flattened parameter vector
        model_dict: Original model dictionary (for keys and device)
        shapes: List of tensor shapes (optional, computed if None)
        sizes: List of tensor sizes (optional, computed if None)
        
    Returns:
        new_model_dict: State dictionary with reconstructed parameters
    """
    if shapes is None or sizes is None:
        W_l = [W.clone().detach() for W in model_dict.values()]
        shapes = [W.shape for W in W_l]
        sizes = [W.numel() for W in W_l]
    
    W_l_new = []
    idx = 0
    for shape, size in zip(shapes, sizes):
        W_l_new.append(theta[idx:idx+size].reshape(shape))
        idx += size
    new_model_dict = {k: v for k, v in zip(model_dict.keys(), W_l_new)}
    return new_model_dict


def cosine_similarity(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
    else:
        raise TypeError("Inputs must be both numpy arrays or both torch tensors")

def vector_angle(a, b):
    return np.rad2deg(np.arccos(cosine_similarity(a, b)))

def get_r_2(X, y):
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return r2_score(y, y_pred)



def get_upper_triangle(matrix):
    """
    Get the upper triangle of a square matrix.
    """
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, k=1)]

import numpy as np
from scipy.linalg import subspace_angles, svd


def compute_covariance(X):
    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X_centered, rowvar=False)
    return cov

def principal_directions(cov, explained_var=0.95):
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    explained_variance_ratio = np.cumsum(eigvals) / np.sum(eigvals)
    k = np.where(explained_variance_ratio > explained_var)[0][0] if explained_variance_ratio[0] < explained_var else 0
    print(k)
    return eigvecs[:, :k+1]

def alignment_score(blob1, blob2, explained_var=0.95):
    # Compute covariance matrices
    cov1 = compute_covariance(blob1)
    cov2 = compute_covariance(blob2)

    # Get top-k principal directions
    dirs1 = principal_directions(cov1, explained_var)
    dirs2 = principal_directions(cov2, explained_var)

    # Compute canonical (principal) angles
    angles = subspace_angles(dirs1, dirs2)

    # Alignment score: cosine of smallest angle (closer to 1 means more aligned)
    score = np.cos(angles[0])
    return score


# def calc_NC1(h, y):
#     with torch.no_grad():
#         classes = torch.unique(y.argmax(1))
#         Sw = 0
#         mean_c = []
#         for c in classes:
#             h_c = h[y.argmax(1) == c]
#             z = h_c - h_c.mean(0).unsqueeze(0)
#             cov = z.unsqueeze(-1) @ z.unsqueeze(1)
#             Sw += cov.sum(0)
#             mean_c.append(h_c.mean(0))
#         Sw /= y.shape[0]
#         M = torch.stack(mean_c).T
#         muG = torch.mean(M, dim=1, keepdim=True)  # CHW 1
#         M_ = M - muG
#         Sb = torch.matmul(M_, M_.T) / len(classes)
#     Sw = Sw.cpu().numpy()
#     Sb = Sb.cpu().numpy()
#     eigvec, eigval, _ = svds(Sb, k=len(classes) - 1)
#     inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
#     return np.trace(Sw @ inv_Sb)

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def calc_NC1(h, y):
    """
    Compute NC1 from hidden activations h and labels y.

    Parameters:
    - h: ndarray of shape (n_samples, hidden_dim), hidden layer activations
    - y: ndarray of shape (n_samples,), class labels

    Returns:
    - nc1: scalar, NC1 value
    """
    h = np.array(h)
    y = np.array(y)
    classes = np.unique(y)
    num_classes = len(classes)
    hidden_dim = h.shape[1]

    # Compute class means and global mean
    class_means = np.zeros((num_classes, hidden_dim))
    global_mean = np.mean(h, axis=0)

    Sw = np.zeros((hidden_dim, hidden_dim))  # within-class covariance
    Sb = np.zeros((hidden_dim, hidden_dim))  # between-class covariance

    for i, cls in enumerate(classes):
        h_cls = h[y == cls]
        mu_cls = np.mean(h_cls, axis=0)
        class_means[i] = mu_cls

        # Within-class scatter
        centered = h_cls - mu_cls
        Sw += centered.T @ centered

    # Between-class scatter
    for i, mu_cls in enumerate(class_means):
        n_i = np.sum(y == classes[i])
        diff = (mu_cls - global_mean).reshape(-1, 1)
        Sb += n_i * (diff @ diff.T)

    trace_Sw = np.trace(Sw)
    trace_Sb = np.trace(Sb)

    nc1 = trace_Sw / trace_Sb if trace_Sb != 0 else np.inf
    return nc1

def calc_PR(h):
    cov = compute_covariance(h)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)


# PCA function using SVD
def pca_torch(X, k=None):
    # Center the data
    X_centered = X - X.mean(dim=0, keepdim=True)

    # SVD decomposition
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    # Select top-k components if specified
    if k is not None:
        S = S[:k]
        Vt = Vt[:k]

    # Compute explained variance
    n_samples = X.shape[0]
    explained_variance = (S ** 2) / (n_samples - 1)

    # Total variance is the sum of variances of all features
    total_var = X_centered.var(dim=0, unbiased=True).sum()
    explained_variance_ratio = explained_variance / total_var

    return {
        'components': Vt,  # shape: (k, n_features)
        'explained_variance': explained_variance.cpu().numpy(),  # shape: (k,)
        'explained_variance_ratio': explained_variance_ratio.cpu().numpy()  # shape: (k,)
    }


def get_all_key_combinations(my_dict):
    # Get keys and lists of values
    keys = list(my_dict.keys())
    values_lists = [my_dict[key] for key in keys]

    # Iterate over all combinations
    combinations = []
    for combination in product(*values_lists):
        current = dict(zip(keys, combination))
        combinations.append(current)  # current is a dict: {key: value, ...}
    return combinations

def factorize_matrix(M, N=None):
    D1, D2 = M.shape
    # Compute full SVD
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    rank = np.sum(S > 1e-10)  # numerical rank

    if N is None:
        N = min(D1,D2)  # default N to rank(M)
    
    # Handle case where N > min(D1,D2)
    if N > min(D1,D2):
        # Pad U with random orthonormal columns
        U_extra = np.zeros([D1, N - min(D1,D2)])
        U = np.hstack([U, U_extra])
        
        # Pad S with zeros
        S = np.pad(S, (0, N - len(S)), mode='constant')
        
        # Pad Vt with random orthonormal rows
        Vt_extra = np.zeros([N - min(D1,D2), D2]) 
        Vt = np.vstack([Vt, Vt_extra])

    # Take N components
    U_N = U[:, :N]
    S_N = np.diag(S[:N])
    Vt_N = Vt[:N, :]

    # Generate random orthogonal matrix
    Q = np.random.randn(N, N)
    Q, _ = np.linalg.qr(Q)
    
    # Create random factorization that still reconstructs M
    A = U_N @ np.sqrt(S_N) @ Q
    B = Q.T @ np.sqrt(S_N) @ Vt_N
    return A, B


def compute_hessian(data_dict, normalize=None):
    x = data_dict['X']; target = data_dict['y']; final_weights = data_dict['final_weights']
    loss_fn = 'CE' if isinstance(data_dict['C'].loss_fn, torch.nn.CrossEntropyLoss) else 'MSE'
    if normalize:
        final_weights = normalize_state_dict(final_weights, normalize)
    # Make sure weights require gradients
    W_l = [w.clone().detach().requires_grad_(True) for w in final_weights.values()]

    # Forward pass through layers
    out = x
    for W in W_l:
        out = out @ W.T  # Linear layer without bias

    # Loss (mean squared error)
    if loss_fn == 'CE':
        # probs = F.softmax(out, dim=1)
        loss = F.cross_entropy(out, target)
    elif loss_fn == 'MSE':
        loss = F.mse_loss(out, target)
    else:
        raise ValueError(f"Invalid loss function: {loss_fn}")

    # Flatten parameters
    params_vector = torch.cat([w.view(-1) for w in W_l])

    # Compute gradients (first-order)
    grads = torch.autograd.grad(loss, W_l, create_graph=True)
    grads_vector = torch.cat([g.view(-1) for g in grads])

    num_params = params_vector.numel()
    hessian = torch.zeros(num_params, num_params)

    for i in range(num_params):
        grad2rd = torch.autograd.grad(grads_vector[i], W_l, retain_graph=True)
        grad2rd_vector = torch.cat([g.contiguous().view(-1) for g in grad2rd])
        hessian[i] = grad2rd_vector

    return hessian


def compute_gradient(data_dict, normalize=None):
    x = data_dict['X']; target = data_dict['y']; final_weights = data_dict['final_weights']
    loss_fn = 'CE' if isinstance(data_dict['C'].loss_fn, torch.nn.CrossEntropyLoss) else 'MSE'
    if normalize:
        final_weights = normalize_state_dict(final_weights, normalize)
    # Make sure weights require gradients
    W_l = [w.clone().detach().requires_grad_(True) for w in final_weights.values()]

    # Forward pass through layers
    out = x
    for W in W_l:
        out = out @ W.T  # Linear layer without bias

    # Loss (mean squared error)
    if loss_fn == 'CE':
        loss = F.cross_entropy(out, target)
    elif loss_fn == 'MSE':
        loss = F.mse_loss(out, target)
    else:
        raise ValueError(f"Invalid loss function: {loss_fn}")

    # Flatten parameters
    params_vector = torch.cat([w.view(-1) for w in W_l])

    # Compute gradients (first-order)
    grads = torch.autograd.grad(loss, W_l, create_graph=True)
    grads_vector = torch.cat([g.view(-1) for g in grads])
    
    return grads_vector

def normalize_W_l(W_l, norm=100):
    if isinstance(W_l[0], torch.Tensor):
        theta = torch.concatenate([W.flatten() for W in W_l])
        factor = torch.linalg.norm(theta)/norm
        return [W/factor for W in W_l]
    else:
        theta = np.concatenate([W.flatten() for W in W_l])
        factor = np.linalg.norm(theta)/norm
    return [W/factor for W in W_l]

def normalize_state_dict(model_dict, norm=100):
    theta, shapes, sizes = state_dict_to_theta(model_dict)
    theta = theta * norm / torch.linalg.norm(theta)
    return theta_to_state_dict(theta, model_dict, shapes, sizes)

def perturb_model_dict(model_dict, direction, norm=1, normalize=None):
    device = next(iter(model_dict.values())).device
    direction = direction.to(device)
    theta, shapes, sizes = state_dict_to_theta(model_dict)
    theta += direction * norm / torch.linalg.norm(direction).to(device)
    if normalize:
        theta = theta * normalize / torch.linalg.norm(theta)
    return theta_to_state_dict(theta, model_dict, shapes, sizes)

def get_loss(data_dict, normalize=None):
    x = data_dict['X']; target = data_dict['y']; final_weights = data_dict['final_weights']
    loss_fn = 'CE' if isinstance(data_dict['C'].loss_fn, torch.nn.CrossEntropyLoss) else 'MSE'
    if normalize:
        final_weights = normalize_state_dict(final_weights, normalize)
    W_effective = reduce(torch.matmul, [W.T for W in final_weights.values()])
    out = x @ W_effective
    if loss_fn == 'CE':
        loss = F.cross_entropy(out, target)
    elif loss_fn == 'MSE':
        loss = F.mse_loss(out, target)
    else:
        raise ValueError(f"Invalid loss function: {loss_fn}")
    return loss.item()

def get_AB(X, w1, w2, b, n):
    # Step 1: Compute target matrix
    Y = (X @ w1) @ w2 + np.ones((X.shape[0], 1)) @ b  # (m, c)

    # Step 2: Compute effective Z = X^\dagger Y
    X_dagger = np.linalg.pinv(X)                     # (d, m)
    Z = X_dagger @ Y                                 # (d, c)

    # Step 3: Low-rank SVD factorization
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    n_max = min(n, min(Z.shape))  # Don't take more components than available
    U_n = np.zeros((U.shape[0], n))  # Initialize with zeros
    S_n = np.zeros((n, n))  # Initialize diagonal matrix with zeros
    Vn = np.zeros((n, Vt.shape[1]))  # Initialize with zeros
    
    # Fill available components
    U_n[:, :n_max] = U[:, :n_max]
    S_n[:n_max, :n_max] = np.diag(np.sqrt(S[:n_max]))
    Vn[:n_max, :] = Vt[:n_max, :]

    A = U_n @ S_n                  # (d, n)
    B = S_n @ Vn                   # (n, c)
    return A, B

def make_synthetic_model_dict(data_dict):
    X_np = data_dict['X'].cpu().numpy()
    y_np = data_dict['y'].cpu().numpy()
    C = data_dict['C']
    L = C.length_corridors[0]
    A = C.max_move
    n_model = 1
    Win = np.concatenate([np.arange(1,L+1, 1), np.arange(-A,A+1)])[:,None]
    Wout = 1/n_model*np.arange(1,L+1, 1)[None,:]**n_model
    b = -1/(n_model+1)*np.arange(1,L+1, 1)[None, :]**(n_model+1)

    W1,W2 = get_AB(X_np, Win, Wout, b, C.hidden_size)
    W2 = torch.tensor(W2.T, dtype=torch.float32)
    W = torch.tensor(W1, dtype=torch.float32)
    W_l = factorize_matrix_to_L_matrices(W, C.L, C.hidden_size)
    model_dict = {k:v for k, v in zip(data_dict['final_weights'].keys(), W_l + [W2])}
    return model_dict


def factorize_matrix_to_L_matrices(W, L, N=None):
    """
    Factorize a matrix W into L matrices with intermediate dimension N.
    
    Args:
        W: Input matrix of shape (m, n)
        L: Number of matrices to factorize into
        N: Intermediate dimension. If None, uses min(m, n)
    
    Returns:
        List of L matrices whose multiplication equals W
    """
    m, n = W.shape
    if N is None:
        N = min(m, n)
    
    # Calculate the scale factor to match W's scale
    # For L matrices, we want their product to have similar scale as W
    # If each matrix has scale s, then the product has scale s^L
    # So we want s^L ≈ scale(W), which means s ≈ scale(W)^(1/L)
    
    # Initialize matrices randomly with appropriate scaling
    matrices = []
    
    # First matrix: shape (m, N)
    matrices.append(torch.randn(N, m, requires_grad=True))
    
    # Middle matrices: shape (N, N)
    for i in range(L - 2):
        matrices.append(torch.randn(N, N, requires_grad=True))
    
    # Last matrix: shape (N, n)
    matrices.append(torch.randn(n, N, requires_grad=True))
    
    with torch.no_grad():
        product = matrices[0].T
        for i in range(1, L):
            product = product @ matrices[i].T
        scale_factor = (torch.norm(W) / torch.norm(product))**(1/L)
        for w in matrices:
            w *= scale_factor
    
    # Optimize to find the factorization
    optimizer = torch.optim.Adam(matrices, lr=0.00001)
    n_steps = 20000
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Compute the product of all matrices
        product = matrices[0].T
        for i in range(1, L):
            product = product @ matrices[i].T

        # Compute loss (MSE between product and target W)
        loss = torch.nn.functional.mse_loss(product, W)
        
        loss.backward()
        optimizer.step()
        
        if step % (n_steps//10) == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
    return matrices

def compute_gradient_np(data_dict, normalize=None):
    def softmax(z):
        z = z - np.max(z)  # For numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def cross_entropy_grad(logits, y_true, num_classes):
        return softmax(logits) - y_true

    def forward_pass(x, weights):
        activations = [x]
        for W in weights:
            x = W @ x
            activations.append(x)
        return activations

    def compute_gradients(x, y, weights):
        L = len(weights)
        num_classes = weights[-1].shape[0]

        # Forward pass
        activations = forward_pass(x, weights)

        # Backward pass
        grads = [None] * L
        delta = cross_entropy_grad(activations[-1], y, num_classes)

        for i in reversed(range(L)):
            a_prev = activations[i].reshape(-1, 1)
            grads[i] = delta.reshape(-1, 1) @ a_prev.T
            if i > 0:
                delta = weights[i].T @ delta

        # Flatten and concatenate all gradients into one vector
        flat_grads = np.concatenate([g.flatten() for g in grads])
        return flat_grads

    W_l = [W.detach().cpu().numpy() for W in data_dict['final_weights'].values()]
    if normalize:
        W_l = normalize_W_l(W_l, normalize)
    X_np = data_dict['X'].cpu().numpy()
    y_np = data_dict['y'].cpu().numpy()
    grad = [compute_gradients(x, y, W_l) for x,y in zip(X_np, y_np)]
    grad = np.array(grad).mean(0)
    return grad


def get_state_dict_norm(model_dict):
    theta, _, _ = state_dict_to_theta(model_dict)
    return torch.linalg.norm(theta)

def grid_search_pert_direction(data_dict, eigs, eigs_v): 
    C = data_dict['C']
    X = data_dict['X']; y = data_dict['y']
    loc_y = data_dict['loc_y']

    C.print_progress = False
    n_samples = 100
    accuracy_map = np.zeros((n_samples, n_samples))
    r_2_map = np.zeros((n_samples, n_samples))
    n_eigs_l = np.linspace(1, len(eigs), n_samples).astype(int)
    norm_l = np.linspace(1, 100, n_samples)
    
    # Pre-compute the sorted eigenvalues and eigenvectors to avoid repeated computation
    sorted_indices = abs(eigs).argsort()
    sorted_eigs_v = torch.real(eigs_v[:, sorted_indices])
    
    # Pre-compute all perturbation directions
    perturbation_directions = []
    for n_eigs in n_eigs_l:
        direction = sorted_eigs_v[:, :n_eigs].mean(axis=1)
        perturbation_directions.append(direction)
    
    for i, (n_eigs, pert_direction) in tqdm(enumerate(zip(n_eigs_l, perturbation_directions))):
        for j, norm in enumerate(norm_l):
            new_weights = perturb_model_dict(data_dict['final_weights'], pert_direction, norm=norm, normalize=1)
            
            W_hidden = get_effective_W_from_model_dict(new_weights, to_hidden=True)
            hidden = X @ W_hidden
            h_np = hidden.cpu().numpy()
            W_output = get_effective_W_from_model_dict(new_weights, to_hidden=False)
            out = X @ W_output
            
            accuracy = (out.argmax(1) == y.argmax(1)).float().mean().item()
            r_2 = get_r_2(PCA(n_components=2).fit_transform(h_np), loc_y)
            accuracy_map[i, j] = accuracy
            r_2_map[i, j] = r_2
    return accuracy_map, r_2_map, norm_l, n_eigs_l

def get_effective_W_from_model_dict(model_dict, to_hidden=False):
    W_l = [W.clone().detach() for W in model_dict.values()]
    W_l = W_l[:-1] if to_hidden else W_l
    W_effective = W_l[0].T
    for W in W_l[1:]:
        W_effective = W_effective @ W.T
    return W_effective